import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.discriminator.model import (NLayerDiscriminator,
                                                weights_init)
from models.modules.losses.lpips import LPIPS


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, lambda_nd=0.0, lambda_mask=0.0,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.lambda_nd = lambda_nd
        self.BCEwithLogit = torch.nn.BCEWithLogitsLoss()
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                            n_layers=disc_num_layers,
                                            use_actnorm=use_actnorm,
                                            ndf=disc_ndf
                                            ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.lambda_mask = lambda_mask

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    def calculate_neigbor_angles(self, points):
        # Assuming depth1 and depth2 are torch tensors with shape (B, C, H, W)
        # Calculate the differences in depth
        # Calculate the differences in x and y
        r_shifted = torch.roll(points, shifts=(0, 1), dims=(2, 3))
        l_shifted = torch.roll(points, shifts=(0, -1), dims=(2, 3))
        x_diff, y_diff = r_shifted[:, 0, :, :] - points[:, 0, :, :], r_shifted[:, 1, :, :] - points[:, 1, :, :]
        r_angle = torch.atan2(y_diff, x_diff)
        x_diff, y_diff = l_shifted[:, 0, :, :] - points[:, 0, :, :], l_shifted[:, 1, :, :] - points[:, 1, :, :]
        l_angle = torch.atan2(y_diff, x_diff)
        return torch.stack((r_angle, l_angle), dim=1)
        # Calculate the angle

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, aug_cls, last_layer=None, cond=None, split="train", points_inputs=None, points_rec=None, mask_logits=None, real_mask=None):
        # now the GAN part
        if optimizer_idx == 0:
            if self.lambda_mask == 0.0 and self.discriminator_weight == 0.0:
                rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean()
            else:
                rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) * real_mask
                rec_loss = (rec_loss.sum(dim=(1, 2, 3)) / real_mask.sum(dim=(1, 2, 3))).mean()
            total_loss = rec_loss

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).mean()
                total_loss += self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])
            ## calculate the angle loss
            real_angle = self.calculate_neigbor_angles(points_inputs)
            fake_angle = self.calculate_neigbor_angles(points_rec)
            nd_loss = torch.abs(real_angle - fake_angle).mean()
            total_loss += self.lambda_nd * nd_loss
            # mask loss
            mask_loss = self.BCEwithLogit(mask_logits, real_mask)
            total_loss += self.lambda_mask * mask_loss

            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(aug_cls(reconstructions.contiguous()))
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((aug_cls(reconstructions.contiguous()), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            d_weight = self.discriminator_weight
            # try:
            #     d_weight = self.calculate_adaptive_weight(total_loss, g_loss, last_layer=last_layer)
            # except RuntimeError:
            #     assert not self.training
            #     d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = total_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"mask": mask_loss.detach(),
                   "q": codebook_loss.detach(),
                   "nd": nd_loss.detach(),
                   "rec": rec_loss.detach(),
                   "p": p_loss.detach(),
                   "d_weight": torch.tensor(d_weight),
                   "disc_factor": torch.tensor(disc_factor),
                   "gan": g_loss.detach(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(aug_cls(inputs.contiguous()).detach())
                logits_fake = self.discriminator(aug_cls(reconstructions.contiguous()).detach())
            else:
                logits_real = self.discriminator(torch.cat((aug_cls(inputs.contiguous()).detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((aug_cls(reconstructions.contiguous()).detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log
