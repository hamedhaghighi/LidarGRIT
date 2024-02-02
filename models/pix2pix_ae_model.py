import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import *


class Pix2PixAEModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt, lidar):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'mask_bce', 'G_nd']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['G']
        # define networks (both generator and discriminator)
        
        opt_m = opt.model
        opt_t = opt.training
        self.eval_metrics = ['cd', 'depth_accuracies', 'depth_errors'] 
        
        if 'depth' in opt_m.modality_B:
            self.visual_names.extend(['synth_depth', 'synth_mask'])
        if 'reflectance' in opt_m.modality_B:
            self.visual_names.extend(['real_reflectance', 'synth_reflectance'])
            self.eval_metrics.append('reflectance_errors')
        input_nc_G = np.array([m2ch[m] for m in opt_m.modality_A]).sum()
        output_nc_G = np.array([m2ch[m] for m in opt_m.out_ch]).sum()
        same_kernel_size = opt.dataset.dataset_A.img_prop.width == opt.dataset.dataset_A.img_prop.height
        self.netG = networks.define_G(input_nc_G, output_nc_G, opt_m.ngf, opt_m.netG, opt_m.norm,
                                      not opt_m.no_dropout, opt_m.init_type, opt_m.init_gain, self.gpu_ids, opt_m.out_ch, same_kernel_size=True)


        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        self.BCEwithLogit = torch.nn.BCEWithLogitsLoss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.lidar = lidar
        self.opt_m = opt_m
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt_t.lr, betas=(opt_t.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

   
    def data_dependent_initialize(self, data):
        return
    def set_seg_model(self, model):
        return

    def set_input(self, data):
        data = fetch_reals(data, self.lidar, self.device)
        for k, v in data.items():
            setattr(self, 'real_' + k, v)
        data_list = []
        for m in self.opt.model.modality_A:
            assert m in data
            data_list.append(data[m])
        self.real_A = torch.cat(data_list, dim=1)
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        out_dict, self.fake_B  = self.netG(self.real_A) # G(A)
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v)
        
    def calculate_neighbor_distances(self, depth_images):
        # Assuming depth_images is a torch tensor with shape (B, C, H, W)
        
        # Shift the tensor to consider differences with all 8 neighbors
        h_diff = depth_images - torch.roll(depth_images, shifts=(0, 1), dims=(2, 3))
        v_diff = depth_images - torch.roll(depth_images, shifts=(1, 0), dims=(2, 3))
        d1_diff = depth_images - torch.roll(depth_images, shifts=(1, 1), dims=(2, 3))
        d2_diff = depth_images - torch.roll(depth_images, shifts=(-1, 1), dims=(2, 3))

        # Concatenate all differences along a new dimension
        diff_tensor = torch.cat((h_diff, v_diff, d1_diff, d2_diff), dim=1)

        # Flatten the tensor to make it one-dimensional
        flat_diff = diff_tensor.view(depth_images.size(0), -1)

        return flat_diff
    
    def calculate_neigbor_angles(self, depth):
        # Assuming depth1 and depth2 are torch tensors with shape (B, C, H, W)
        # Calculate the differences in depth
        points = self.lidar.depth_to_xyz(tanh_to_sigmoid(depth))
        # Calculate the differences in x and y
        r_shifted = torch.roll(points, shifts=(0, 1), dims=(2, 3))
        l_shifted = torch.roll(points, shifts=(0, -1), dims=(2, 3))
        x_diff, y_diff = r_shifted[:, 0, :, :] - points[:, 0, :, :], r_shifted[:, 1, :, :] - points[:, 1, :, :]
        r_angle = torch.atan2(y_diff, x_diff)
        x_diff, y_diff = l_shifted[:, 0, :, :] - points[:, 0, :, :], l_shifted[:, 1, :, :] - points[:, 1, :, :]
        l_angle = torch.atan2(y_diff, x_diff)
        return torch.stack((r_angle, l_angle), dim=1)
        # Calculate the angle

    
    def calc_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A)
        fake_angle = self.calculate_neigbor_angles(self.fake_B)
        real_angle = self.calculate_neigbor_angles(self.real_A)
        self.loss_G_nd = self.criterionL1(fake_angle, real_angle)
        self.loss_mask_bce = self.BCEwithLogit(self.synth_mask_logit, self.real_mask) if self.opt.model.lambda_mask > 0.0 else 0.0
        
        self.loss_G = self.loss_G_L1 * self.opt.model.lambda_L1\
              + self.loss_mask_bce * self.opt.model.lambda_mask + self.loss_G_nd * self.opt.model.lambda_nd
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.calc_loss_G()
        self.loss_G.backward()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
