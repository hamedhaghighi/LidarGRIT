import time
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from fid import FID
from dataset.datahandler import get_data_loader
from rangenet.tasks.semantic.modules.segmentator import *
import yaml
import argparse
import numpy as np
import torch
import tqdm
import os
from util.lidar import LiDAR
from util import *
from collections import defaultdict
import shutil
from util.sampling.fps import downsample_point_clouds
from util.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from util.metrics.jsd import compute_jsd
from util.metrics.swd import compute_swd
from util.metrics.seg_accuracy import compute_seg_accuracy

import random


os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class M_parser():
    def __init__(self, cfg_path, data_dir, data_dir_B, load):
        opt_dict = yaml.safe_load(open(cfg_path, 'r'))
        dict_class = make_class_from_dict(opt_dict)
        members = [attr for attr in dir(dict_class) if not callable(getattr(dict_class, attr)) and not attr.startswith("__")]
        for m in members:
            setattr(self, m, getattr(dict_class, m))
        if data_dir != '':
            self.dataset.dataset_A.data_dir = data_dir
        if data_dir_B != '':
            self.dataset.dataset_B.data_dir = data_dir_B
        self.training.test = True
        self.model.isTrain = self.training.isTrain = not self.training.test
        self.training.epoch_decay = self.training.n_epochs//2



def modify_opt_for_fast_test(opt):
    opt.n_epochs = 2
    opt.epoch_decay = opt.n_epochs//2
    opt.display_freq = 1
    opt.print_freq = 1
    opt.save_latest_freq = 1
    opt.max_dataset_size = 10
    opt.batch_size = 2


def check_exp_exists(opt, cfg_args):
    cfg_path = cfg_args.cfg
    opt_t = opt.training
    opt_m = opt.model
    opt_d = opt.dataset.dataset_A
    modality_A = '_'.join(opt_m.modality_A)
    if hasattr(opt_m, 'modality_cond'):
        cond_modality = '_'.join(opt_m.modality_cond)
    out_ch = '_'.join(opt_m.out_ch)
    if cfg_args.load != '':
        # opt_t.name = cfg_path.split(os.sep)[1]
        opt_t.name = cfg_args.load
    elif cfg_args.fast_test:
        opt_t.name = 'test'
    else:
        if 'pix2pix' in opt_m.name:
            opt_t.name = f'pix2pix_modality_A_{modality_A}_out_ch_{out_ch}_L_L1_{opt_m.lambda_L1}_L_nd_{opt_m.lambda_nd}' \
                + f'_L_GAN_{opt_m.lambda_LGAN}_L_mask_{opt_m.lambda_mask}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}'
        elif 'vqgan' in opt_m.name:
            losscfg = opt_m.vqmodel.lossconfig.params
            opt_t.name = f'vqgan_modality_A_{modality_A}_out_ch_{out_ch}_L_nd_{losscfg.lambda_nd}_L_disc_{losscfg.disc_weight}' \
                + f'_L_mask_{opt_m.lambda_mask}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}_bs_{opt_t.batch_size}'
        elif 'transformer' in opt_m.name:
            opt_t.name = f'transformer_modality_A_{modality_A}_out_ch_{out_ch}' \
                + f'_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}'
        
    exp_dir = os.path.join(opt_t.checkpoints_dir, opt_t.name)
    if not opt_t.continue_train and opt_t.isTrain:
        if os.path.exists(exp_dir):
            reply = ''
            # raise Exception('Checkpoint exists!!')
            while not reply.startswith('y') and not reply.startswith('n'):
                reply = str(input(f'exp_dir {exp_dir} exists. Do you want to delete it? (y/n): \n')).lower().strip()
            if reply.startswith('y'):
                shutil.rmtree(exp_dir)
            else:
                print('Please Re-run the program with \"continue train\" enabled')
                exit(0)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(cfg_path, exp_dir)
    else:
        assert os.path.exists(exp_dir)

def main(runner_cfg_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path of the config file')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the dataset A')
    parser.add_argument('--data_dir_B', type=str, default='', help='Path of the dataset B')
    parser.add_argument('--seg_cfg_path', type=str, default='', help='Path of segmentator cfg')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--norm_label', action='store_true', help='normalise labels')
    parser.add_argument('--load', type=str, default='', help='the name of the experiment folder while loading the experiment')
    parser.add_argument('--ref_dataset_name', type=str, default='', help='reference dataset name for measuring unsupervised metrics')
    parser.add_argument('--on_input', action='store_true', help='unsupervised metrics is computerd on dataset A')
    parser.add_argument('--no_inv', action='store_true', help='use it to calc unsupervised metrics on input inv, in case modality_B does not contain inv')
    parser.add_argument('--on_real', action='store_true', help='use it to calc unsupervised metrics on input inv, in case modality_B does not contain inv')
    
    cl_args = parser.parse_args()
    if runner_cfg_path is not None:
        cl_args.cfg = runner_cfg_path
    if 'checkpoints' in cl_args.cfg:
        cl_args.load = cl_args.cfg.split(os.path.sep)[1]
    split = 'train/val'
    if cl_args.ref_dataset_name == 'semanticPOSS':
        seqs = [0, 0, 5] if not cl_args.fast_test else [0, 0 , 0]
        ids = [75, 385, 200] if not cl_args.fast_test else [1, 2, 3]
    else:
        seqs = [0, 0,1, 1, 2, 5] if not cl_args.fast_test else [0, 0 , 0]
        ids = [1, 268,237, 158, 345, 586] if not cl_args.fast_test else [1, 2, 3]
    if cl_args.on_real:
        if cl_args.ref_dataset_name == 'semanticPOSS':
            seqs = [4, 00, 00] 
            ids = [141, 450, 475]
        else:
            seqs = [0, 2,  10, 5] 
            ids = [3309, 229, 378, 2041]
    opt = M_parser(cl_args.cfg, cl_args.data_dir, cl_args.data_dir_B, cl_args.load)
    if cl_args.on_real:
        opt.dataset.dataset_A.name = cl_args.ref_dataset_name
    opt.model.norm_label = cl_args.norm_label
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)
    random.seed(opt.training.seed)
        
    # DATA = yaml.safe_load(open(pa.cfg_dataset, 'r'))
    ## test whole code fast
    if cl_args.fast_test and opt.training.isTrain:
        modify_opt_for_fast_test(opt.training)
    if not opt.training.isTrain:
        opt.training.n_epochs = 1
    check_exp_exists(opt, cl_args)
    device = torch.device('cuda:{}'.format(opt.training.gpu_ids[0])) if opt.training.gpu_ids else torch.device('cpu') 
    ds_cfg = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{opt.dataset.dataset_A.name}_cfg.yml', 'r')))
    if not hasattr(opt.dataset.dataset_A, 'data_dir'):
        opt.dataset.dataset_A.data_dir = ds_cfg.data_dir
    ds_cfg_ref = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{cl_args.ref_dataset_name}_cfg.yml', 'r')))
    lidar_A = LiDAR(
    cfg=ds_cfg,
    height=opt.dataset.dataset_A.img_prop.height,
    width=opt.dataset.dataset_A.img_prop.width).to(device)
    lidar_ref = LiDAR(
    cfg=ds_cfg_ref,
    height=opt.dataset.dataset_A.img_prop.height,
    width=opt.dataset.dataset_A.img_prop.width).to(device)
    lidar = lidar_ref
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    g_steps = 0
    ignore_label = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]

    is_ref_semposs = cl_args.ref_dataset_name == 'semanticPOSS'
    val_dl, val_dataset = get_data_loader(opt, split, opt.training.batch_size, shuffle=False, is_ref_semposs=is_ref_semposs)
    data_list = val_dataset.datalist
    dataset_A_datalist = np.array(data_list)
    dataset_A_selected_idx = []
    for seq, id in zip(seqs, ids):
        pcl_file_path = os.path.join(ds_cfg.data_dir, 'sequences', str(seq).zfill(2), 'velodyne', str(id).zfill(6)+('.bin' if ds_cfg.is_raw else '.npy'))
        dataset_A_selected_idx.append(np.where(dataset_A_datalist == pcl_file_path)[0][0])
    
    with torch.no_grad():
        seg_model = Segmentator(dataset_name=cl_args.ref_dataset_name if cl_args.seg_cfg_path == '' else 'synth', cfg_path=cl_args.seg_cfg_path).to(device)
    model = create_model(opt, lidar_A, None)      # create a model given opt.model and other options
    model.set_seg_model(seg_model)               # regular setup: load and print networks; create schedulers
    ## initilisation of the model for netF in cut
    val_dl_iter = iter(val_dl); data = next(val_dl_iter); model.data_dependent_initialize(data)
    model.setup(opt.training)
    # n_test_batch = 2 if cl_args.fast_test else  len(test_dl)
    # test_dl_iter = iter(test_dl)
    data_dict = defaultdict(list)
    # N = 2 * opt.training.batch_size if cl_args.fast_test else min(len(test_dataset), len(val_dataset), 1000)
    start_from_epoch = model.schedulers[0].last_epoch if opt.training.continue_train else 0 
    val_dl_iter = iter(val_dl)
    n_val_batch = 2 if cl_args.fast_test else  len(val_dl)
    ##### validation
    val_losses = defaultdict(list)
    model.train(False)
    tag = 'val' if opt.training.isTrain else 'test'
    val_tq = tqdm.tqdm(total=len(dataset_A_selected_idx), desc='val_Iter', position=5)
    for i, idx in enumerate(dataset_A_selected_idx):
        data = val_dataset[idx]
        for k ,v in data.items():
            if k != 'path':
                data[k] = v.unsqueeze(0)
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        fetched_data = fetch_reals(data, lidar_A, device, opt.model.norm_label)
        if cl_args.on_input:
            # assert is_two_dataset == False
            if 'depth' in fetched_data:
                synth_depth = fetched_data['depth']
            if 'reflectance' in fetched_data:
                synth_reflectance = fetched_data['reflectance']
            if 'mask' in fetched_data:
                synth_mask = fetched_data['mask']
        else:
            if hasattr(model, 'synth_reflectance'):
                synth_reflectance = model.synth_reflectance 
            if hasattr(model, 'synth_mask'):
                synth_mask = model.synth_mask
            if hasattr(model, 'synth_depth') and not cl_args.no_inv:
                synth_depth = model.synth_depth
            else:
                synth_depth = fetched_data['depth'] * synth_mask
        current_visuals = model.get_current_visuals()
        if hasattr(model, 'synth_reflectance'):
            synth_depth = lidar.revert_depth(tanh_to_sigmoid(synth_depth), norm=False)
            synth_points = lidar.depth_to_xyz(tanh_to_sigmoid(synth_depth)) * lidar.max_depth
            synth_reflectance = tanh_to_sigmoid(synth_reflectance)
            synth_data = torch.cat([synth_depth, synth_points, synth_reflectance, synth_mask], dim=1)
            pred, _ = seg_model(synth_data * fetched_data['mask'])
            pred = pred.argmax(dim=1)
            current_visuals['synth_label'] = pred
        seq = seqs[i]
        _id = ids[i]
        # if is_two_dataset:
        visualizer.display_current_results('',current_visuals, [seq, _id, cl_args.on_input, cl_args.on_real],ds_cfg, opt.dataset.dataset_A.name, lidar_A, ds_cfg_ref,\
                cl_args.ref_dataset_name ,lidar, save_img=True)
        # else:
            # visualizer.display_current_results('', current_visuals, (seq, id),ds_cfg, opt.dataset.dataset_A.name, lidar, save_img=True)
        val_tq.update(1)


if __name__ == '__main__':
    main()
    
 