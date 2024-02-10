import torch
import numpy as np
from .base_model import BaseModel
from util import *
from models.vqgan import VQModel
from models.modules.transformer.mingpt import GPT
from models.util import init_net
from models.modules.losses.lpips import LPIPS
from models.modules.util import SOSProvider
from util import class_to_dict
from models.util import instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class TransformerModel(BaseModel):
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
        self.loss_names = ['t']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['Transformer']
        # define networks (both generator and discriminator)
        
        opt_m = opt.model
        opt_t = opt.training
        self.eval_metrics = ['cd', 'depth_accuracies', 'depth_errors'] 
        self.be_unconditional = True
        self.sos_token = 0
        self.learning_rate = opt_t.lr * opt_t.batch_size

        if 'depth' in opt_m.modality_B:
            self.visual_names.extend(['rec_synth_depth', 'synth_depth'])

        if 'reflectance' in opt_m.modality_B:
            self.visual_names.extend(['real_reflectance', 'synth_reflectance'])
            self.eval_metrics.append('reflectance_errors')

        input_nc_G = np.array([m2ch[m] for m in opt_m.modality_A]).sum()
        output_nc_G = np.array([m2ch[m] for m in opt_m.out_ch]).sum()
        input_nc_D = np.array([m2ch[m] for m in opt_m.modality_B]).sum()
        # same_kernel_size = opt.dataset.dataset_A.img_prop.width == opt.dataset.dataset_A.img_prop.height
        opt_m_dict = class_to_dict(opt_m)
        model = VQModel(**opt_m_dict['first_stage_config'], out_ch=opt_m.out_ch).to(self.gpu_ids[0] if len(self.gpu_ids) > 0 else 'cpu')
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        cond_stage_config = opt_m_dict['cond_stage_config']

        if cond_stage_config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif cond_stage_config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(cond_stage_config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

        self.netTransformer = init_net(GPT(**opt_m_dict['transformer_config']), self.gpu_ids)
        self.permuter = None
        self.downsample_cond_size = -1
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.lidar = lidar
        self.opt_m = opt_m
        self.pkeep = 1.0
        if self.isTrain:
            self.optimizer = self.configure_optimizers()
            self.optimizers.append(self.optimizer)

   
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

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    
    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.netTransformer.module.get_block_size()
        # assert not self.netTransformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.netTransformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.netTransformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        if self.permuter is not None:
            indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        if self.permuter is not None:
            index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, quant_z, z_indices, c_indices, temperature=None, top_k=None, callback=None, **kwargs):



        # create a "half"" sample
        # z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        # index_sample = self.sample(z_start_indices, c_indices,
        #                            steps=z_indices.shape[1]-z_start_indices.shape[1],
        #                            temperature=temperature if temperature is not None else 1.0,
        #                            sample=True,
        #                            top_k=top_k if top_k is not None else 100,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample = self.decode_to_img(index_sample, quant_z.shape)


        # # det sample
        # z_start_indices = z_indices[:, :0]
        # index_sample = self.sample(z_start_indices, c_indices,
        #                            steps=z_indices.shape[1],
        #                            sample=False,
        #                            callback=callback if callback is not None else lambda k: None)
        # x_sample_det = self.decode_to_img(index_sample, quant_z.shape)
        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        out_dict, _ = self.decode_to_img(index_sample, quant_z.shape)
        for k , v in out_dict.items():
            setattr(self, 'synth_' + k , v)

        # reconstruction
        out_dict, _ = self.decode_to_img(z_indices, quant_z.shape)
        for k , v in out_dict.items():
            setattr(self, 'rec_synth_' + k , v)

    

    def forward(self):
        # one step to produce the logits
        quant_z, z_indices = self.encode_to_z(self.real_A)
        _, c_indices = self.encode_to_c(self.real_A)

        if self.isTrain and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.netTransformer.module.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        self.log_images(quant_z, z_indices, c_indices)

        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.netTransformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target
        
    def optimize_parameters(self):
        logits, target = self.forward()            
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.loss_t = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        setattr(self, 'loss_t', self.loss_t)
        self.loss_t.backward()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights
        self.global_step += 1

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.netTransformer.module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.netTransformer.module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer