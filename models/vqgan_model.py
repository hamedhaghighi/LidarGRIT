import torch
import numpy as np
from .base_model import BaseModel
from util import *
from models.modules.diffusionmodules.model import Encoder, Decoder
from models.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from models.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from models.modules.discriminator.model import NLayerDiscriminator
from models.modules.losses.lpips import LPIPS
from models.vqgan import VQModel
from util import class_to_dict, diff_augment, SphericalOptimizer
from models.util import init_net
from tqdm import tqdm

class VQGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    global_step = 0
    def __init__(self, opt, lidar):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['total_G', 'q', 'rec', 'p', 'gan', 'nd', 'mask', 'disc', 'd_weight']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['VQ']
        
        opt_m = opt.model
        opt_t = opt.training
        self.eval_metrics = ['cd', 'depth_accuracies', 'depth_errors', 'val_nd', 'val_rec'] 
        self.visual_names.extend(['real_depth_aug'])
        if 'depth' in opt_m.modality_B:
            self.visual_names.extend(['synth_depth', 'synth_mask'])
        if 'reflectance' in opt_m.modality_B:
            self.visual_names.extend(['real_reflectance', 'synth_reflectance'])
            self.eval_metrics.append('reflectance_errors')

        input_nc_G = np.array([m2ch[m] for m in opt_m.modality_A]).sum()
        output_nc_G = np.array([m2ch[m] for m in opt_m.out_ch]).sum()
        input_nc_D = np.array([m2ch[m] for m in opt_m.modality_B]).sum()
        # same_kernel_size = opt.dataset.dataset_A.img_prop.width == opt.dataset.dataset_A.img_prop.height
        opt_m_dict = class_to_dict(opt_m)
        self.netVQ = init_net(VQModel(**opt_m_dict['vqmodel'], out_ch=opt_m.out_ch), self.gpu_ids)
        self.Aug = diff_augment.DiffAugment(opt_m.augment)
        self.lidar = lidar
        self.opt_m = opt_m
        if self.isTrain:
            self.netVQ.module.learning_rate = opt_t.lr * opt_t.batch_size
            self.optimizers = self.netVQ.module.configure_optimizers()


   
    def data_dependent_initialize(self, data):
        return
    def set_seg_model(self, model):
        return

    def set_input(self, data):
        data = fetch_reals(data, self.lidar, self.device)
        for k, v in data.items():
            setattr(self, 'real_' + k, v)
        setattr(self, 'real_depth_aug', self.Aug(data['depth']))
        data_list = []
        for m in self.opt.model.modality_A:
            assert m in data
            data_list.append(data[m])
        self.real_A = torch.cat(data_list, dim=1)

        data_list = []
        for m in self.opt.model.modality_B:
            assert m in data
            data_list.append(data[m])
        self.real_B = torch.cat(data_list, dim=1)
        
    
    def forward(self, train=True):
        """Run forward pass; called by both functions <optimize_parameters> and <vallidatee>."""
        out_dict, self.fake_B, qloss = self.netVQ(self.Aug(self.real_A) if train else self.real_A)
        self.qloss = qloss
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v)
        
    def reconstruct(self):
        num_step = 100
        perturb_latent = True
        noise_ratio = 0.75
        noise_sigma = 1.0
        lr_rampup_ratio = 0.05
        lr_rampdown_ratio = 0.25
        def lr_schedule(iteration):
            t = iteration / num_step
            gamma = min(1.0, (1.0 - t) / lr_rampdown_ratio)
            gamma = 0.5 - 0.5 * np.cos(gamma * np.pi)
            gamma = gamma * min(1.0, t / lr_rampup_ratio)
            return gamma
        
        self.set_requires_grad(self.netVQ, False)
        B, _, H, W = self.real_A.shape
        symmetric = self.opt_m.vqmodel.ddconfig.symmetric
        l  = len(self.opt_m.vqmodel.ddconfig.ch_mult) - 1
        z_dim_h, z_dim_w = (H//2**l, W//2**l) if symmetric else (H//2**(l//2 + l%2), W//2**l)
        z_dim_c = self.opt_m.vqmodel.embed_dim
        
        latent = torch.randn(B, z_dim_c, z_dim_h, z_dim_w, device=self.device)
        latent.div_(latent.pow(2).mean(dim=[1,2,3], keepdim=True).add(1e-9).sqrt())
        latent = torch.nn.Parameter(latent).requires_grad_()

        optim = SphericalOptimizer(params=[latent], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_schedule)

        # optimize the latent
        for current_step in tqdm(range(num_step), leave=False):
            progress = current_step / num_step
            # noise
            w = max(0.0, 1.0 - progress / noise_ratio)
            noise_strength = 0.05 * noise_sigma * w ** 2
            noise = noise_strength * torch.randn_like(latent)
            # forward G
            quant, emb_loss, _ = self.netVQ.module.quantize(latent + noise if perturb_latent else latent)
            out_dict , out = self.netVQ.module.decode(quant)
            for k , v in out_dict.items():
                setattr(self, 'synth_' + k , v)
            loss, loss_G_dict = self.netVQ.module.training_step(self.real_A, out, 0, global_step=0,\
                                                                            aug_cls=self.Aug, qloss=emb_loss, lidar=self.lidar, mask_logits=self.synth_mask_logit, real_mask=self.real_mask)
            if current_step % 10 == 0: 
                print_msg = f'Step: {current_step}'
                for k , v in loss_G_dict.items():
                    print_msg +=  f' {k} : {int(v.item()* 10000)/10000}'
                print(print_msg)
                print_msg += '\n'
            # per-sample gradients
            optim.zero_grad()
            loss.backward(gradient=torch.ones_like(loss))
            optim.step()
            scheduler.step()

        return
    
    @torch.no_grad()
    def validate(self):
        self.forward(train=False)
        _, loss_G_dict = self.netVQ.module.training_step(self.real_A, self.fake_B, 0, global_step=self.global_step,\
                                                                                  aug_cls=self.Aug, qloss=self.qloss, lidar=self.lidar, mask_logits=self.synth_mask_logit, real_mask=self.real_mask)
        for k, v in loss_G_dict.items():
            setattr(self, 'val_' + k, v.item())

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.optimizers[1].zero_grad()     # set D's gradients to zero
        self.loss_D, loss_D_dict = self.netVQ.module.training_step(self.real_A, self.fake_B, 1, global_step=self.global_step, aug_cls=self.Aug)
        for k, v in loss_D_dict.items():
            setattr(self, 'loss_' + k, v)
        self.loss_D.backward()                # calculate gradients for D
        self.optimizers[1].step()          # update D's weights
        # update G
        self.optimizers[0].zero_grad()     # set G's gradients to zero
        self.loss_total_G, loss_G_dict = self.netVQ.module.training_step(self.real_A, self.fake_B, 0, global_step=self.global_step,\
                                                                                  aug_cls=self.Aug, qloss=self.qloss, lidar=self.lidar, mask_logits=self.synth_mask_logit, real_mask=self.real_mask)
        for k, v in loss_G_dict.items():
            setattr(self, 'loss_' + k, v)
        self.loss_total_G.backward()                # calculate gradients for G
        self.optimizers[0].step()          # update G's weights
        self.global_step += 1