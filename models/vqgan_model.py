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
from util import class_to_dict
from models.util import init_net

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
        self.loss_names = ['total_G', 'D', 'q', 'rec', 'p', 'gan', 'nd']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['VQ']

        
        opt_m = opt.model
        opt_t = opt.training
        self.eval_metrics = ['cd', 'depth_accuracies', 'depth_errors', 'val_nd', 'val_rec'] 
        
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
        
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        out_dict, self.fake_B, _ = self.netVQ(self.real_A)
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v)
        
    @torch.no_grad()
    def validate(self):
        self.forward()
        points_input = self.lidar.depth_to_xyz(tanh_to_sigmoid(self.real_A))
        points_rec = self.lidar.depth_to_xyz(tanh_to_sigmoid(self.fake_B))
        _, loss_G_dict = self.netVQ.module.training_step(self.real_A, 0, points_inputs=points_input, \
                                                                          points_rec=points_rec, global_step=self.global_step)
        for k, v in loss_G_dict.items():
            setattr(self, 'val_' + k, v.item())

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        points_input = self.lidar.depth_to_xyz(tanh_to_sigmoid(self.real_A))
        points_rec = self.lidar.depth_to_xyz(tanh_to_sigmoid(self.fake_B))
        # update D
        self.optimizers[1].zero_grad()     # set D's gradients to zero
        self.loss_D, _ = self.netVQ.module.training_step(self.real_A, 1, global_step=self.global_step)
        self.loss_D.backward()                # calculate gradients for D
        self.optimizers[1].step()          # update D's weights
        # update G
        self.optimizers[0].zero_grad()     # set G's gradients to zero
        self.loss_total_G, loss_G_dict = self.netVQ.module.training_step(self.real_A, 0, points_inputs=points_input, \
                                                                          points_rec=points_rec, global_step=self.global_step)
        for k, v in loss_G_dict.items():
            setattr(self, 'loss_' + k, v)
        self.loss_total_G.backward()                # calculate gradients for G
        self.optimizers[0].step()          # update G's weights
        self.global_step += 1