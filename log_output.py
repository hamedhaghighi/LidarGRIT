import argparse
import hashlib
import os
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import tqdm
import yaml

from dataset.datahandler import get_data_loader
# from data import create_dataset
from models import create_model
from rangenet.tasks.semantic.modules.segmentator import *
from util import *
from util.lidar import LiDAR
from util.metrics.cov_mmd_1nna import compute_cd
from util.metrics.depth import compute_depth_accuracy, compute_depth_error
from util.visualizer import Visualizer

os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def calc_compression_metrics(points_ref, points_gen):
    points_ref = points_ref[~torch.all(points_ref == 0, dim=1)].cpu().numpy()
    points_gen = points_gen[~torch.all(points_gen == 0, dim=1)].cpu().numpy()
    # points_ref = points_ref.cpu().numpy()
    # points_gen = points_gen.cpu().numpy()
    return pcerror(points_ref,points_gen,None,'-r 1023',None)


def calc_supervised_metrics(synth_depth, real_depth, real_mask, real_points, lidar):
    # self.forward()
    points_gen = lidar.depth_to_xyz(tanh_to_sigmoid(synth_depth))
    points_gen = flatten(points_gen)
    points_ref = flatten(real_points)
    depth_ref = lidar.denormalize_depth(tanh_to_sigmoid(real_depth))
    depth_gen = lidar.denormalize_depth(tanh_to_sigmoid(synth_depth))
    cd = compute_cd(points_ref, points_gen).mean().item()
    MAE = (torch.abs(depth_ref - depth_gen) * real_mask).sum(dim=(1, 2, 3))/ real_mask.sum(dim=(1, 2, 3))
    MAE = MAE.item()
    scores = {'cd': cd, 'MAE': MAE}
    accuracies = compute_depth_accuracy(depth_ref, depth_gen, real_mask)
    depth_accuracies = {'depth/' + k: v.mean().item() for k ,v in accuracies.items()}
    errors = compute_depth_error(depth_ref, depth_gen, real_mask)
    depth_errors = {'depth/' + k: v.mean().item() for k ,v in errors.items()}
    scores.update(depth_accuracies)
    scores.update(depth_errors)
    scores['p2p-PSNR'] = calc_compression_metrics(points_ref.squeeze() * lidar.max_depth,points_gen.squeeze()* lidar.max_depth)
    # if 'reflectance' in self.opt.model.modality_B:
    #     reflectance_ref = tanh_to_sigmoid(self.real_reflectance) + 1e-8
    #     reflectance_gen = tanh_to_sigmoid(self.rec_synth_reflectance if is_transformer else self.synth_reflectance) + 1e-8
    #     errors = compute_depth_error(reflectance_ref, reflectance_gen)
    #     self.reflectance_errors = {'reflectance/' + k: v.mean().item() for k ,v in errors.items()}
    #     self.reflectance_ssim = self.crterionSSIM(self.real_reflectance, self.synth_reflectance, torch.ones_like(self.real_reflectance))
    return scores

class M_parser():
    def __init__(self, cfg_path, data_dir, is_test):
        opt_dict = yaml.safe_load(open(cfg_path, 'r'))
        dict_class = make_class_from_dict(opt_dict)
        members = [attr for attr in dir(dict_class) if not callable(getattr(dict_class, attr)) and not attr.startswith("__")]
        for m in members:
            setattr(self, m, getattr(dict_class, m))
        if data_dir != '':
            self.dataset.dataset_A.data_dir = data_dir
        self.training.test = is_test
        self.model.isTrain = self.training.isTrain = not self.training.test
        self.training.epoch_decay = self.training.n_epochs//2
        if 'transformer' in self.model.name:
            vqckpt_dir = self.model.vq_ckpt_path.split(os.path.sep)[:-1]
            vqconfig_path = os.path.join(os.path.sep.join(vqckpt_dir), 'vqgan.yaml')
            vqcfg_dict = yaml.safe_load(open(vqconfig_path, 'r'))
            self.model.modality_A = vqcfg_dict['model']['modality_A']
            self.model.modality_B = vqcfg_dict['model']['modality_B']
            self.model.out_ch = vqcfg_dict['model']['out_ch']
            self.model.transformer_config.vocab_size = vqcfg_dict['model']['vqmodel']['n_embed']
            H , W = vqcfg_dict['dataset']['dataset_A']['img_prop']['height'], vqcfg_dict['dataset']['dataset_A']['img_prop']['width']
            symmetric = vqcfg_dict['model']['vqmodel']['ddconfig']['symmetric']
            l  = len(vqcfg_dict['model']['vqmodel']['ddconfig']['ch_mult']) - 1
            self.model.transformer_config.block_size = (H//2**l) * (W//2**l) if symmetric else (H//2**(l//2 + l%2)) * (W//2**l)


def hash_string(input_string):
    # Encode the input string as bytes
    input_bytes = input_string.encode('utf-8')
    
    # Hash the bytes using SHA-256
    hashed_bytes = hashlib.sha256(input_bytes)
    
    # Get the hexadecimal representation of the hash
    hashed_string = hashed_bytes.hexdigest()
    
    return hashed_string

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
    out_ch = '_'.join(opt_m.out_ch)
    if cfg_args.load != '':
        opt_t.name = cfg_args.load
    elif cfg_args.fast_test:
        opt_t.name = 'test_trans' if 'transformer' in opt.model.name else 'test'
    else:
        opt_t.name = hash_string(yaml.dump(class_to_dict(opt)))
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
    parser.add_argument('--seg_cfg_path', type=str, default='', help='Path of segmentator cfg')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--norm_label', action='store_true', help='normalise labels')
    parser.add_argument('--ref_dataset_name', type=str, default='', help='reference dataset name for measuring unsupervised metrics')
    parser.add_argument('--on_input', action='store_true', help='unsupervised metrics is computerd on dataset A')
    parser.add_argument('--no_inv', action='store_true', help='use it to calc unsupervised metrics on input inv, in case modality_B does not contain inv')
    parser.add_argument('--completion', action='store_true', help='use it to complete the distorted point clouds')
    parser.add_argument('--compression', action='store_true', help='use it to calc compression metrics')
    parser.add_argument('--calc_mae', action='store_true', help='use it to complete the distorted point clouds')
    N = 100
    cl_args = parser.parse_args()
    if runner_cfg_path is not None:
        cl_args.cfg = runner_cfg_path
    if 'checkpoints' in cl_args.cfg:
        cl_args.load = cl_args.cfg.split(os.path.sep)[1]
    split = 'test'
    if cl_args.ref_dataset_name == 'kitti':
        # seqs = [11, 12 ,13, 11, 12, 13] if not cl_args.fast_test else [11, 11 , 11]
        # ids = [1, 268,237, 158, 200, 100] if not cl_args.fast_test else [1, 2, 3]
        seqs = [random.randint(11, 21) for _ in range(N)]
        ids = [random.randint(0, 400) for _ in range(N)]
    elif cl_args.ref_dataset_name == 'kitti_360':
        seqs = [0] * (N//2)
        seqs.extend([2] * (N//2))
        ids = [random.randint(0, 11517) for _ in range(N//2)]
        ids.extend([random.randint(4391, 19239) for _ in range(N//2)])
    opt = M_parser(cl_args.cfg, cl_args.data_dir, True)
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

    test_dl, test_dataset = get_data_loader(opt, split, opt.training.batch_size, shuffle=False)
    data_list = test_dataset.datalist
    dataset_A_datalist = np.array(data_list)
    dataset_A_selected_idx = []
    for seq, id in zip(seqs, ids):
        if cl_args.ref_dataset_name == 'kitti_360': 
            pcl_file_path = os.path.join(ds_cfg.data_dir, f'2013_05_28_drive_{seq:04d}_sync/velodyne_points/data/', str(id).zfill(10)+('.bin' if ds_cfg.is_raw else '.npy'))
        else:
            pcl_file_path = os.path.join(ds_cfg.data_dir, 'sequences', str(seq).zfill(2), 'velodyne', str(id).zfill(6)+('.bin' if ds_cfg.is_raw else '.npy'))
        dataset_A_selected_idx.append(np.where(dataset_A_datalist == pcl_file_path)[0][0])
    with torch.no_grad():
        seg_model = Segmentator(dataset_name=cl_args.ref_dataset_name if cl_args.seg_cfg_path == '' else 'synth', cfg_path=cl_args.seg_cfg_path).to(device)
    
    model = create_model(opt, lidar_A, None)      # create a model given opt.model and other options
    model.set_seg_model(seg_model)               # regular setup: load and print networks; create schedulers
    ## initilisation of the model for netF in cut
    test_dl_iter = iter(test_dl); data = next(test_dl_iter); model.data_dependent_initialize(data)
    model.setup(opt.training)
    test_dl_iter = iter(test_dl)
    ##### validation
    model.train(False)
    tq = tqdm.tqdm(total=len(dataset_A_selected_idx), desc='val_Iter', position=5)
    t_scores = defaultdict(list)
    for i, idx in enumerate(dataset_A_selected_idx):
        
        data = test_dataset[idx]
        if cl_args.completion:
            _, H, W = data['depth'].shape
            mask = torch.zeros_like(data['depth']).repeat_interleave(4, dim=0)
            mask[0, ...] = 1
            mask[1, ::4, :] = 1  # 25% beams
            mask[2, ::8, :] = 1  # 1/8 beams 
            # mask[2, :] = torch.empty(H, 1).bernoulli_(0.5)  # random 50% beams
            mask[3, :] = torch.empty(H, W).bernoulli_(0.1)  # random 10% points

            data['depth'] = mask * data['depth']
            data['mask'] = mask * data['mask']
            data['points'] = mask[:, None].repeat_interleave(3, dim=1) * data['points']
        for j in range(4 if cl_args.completion else 1):
            if cl_args.calc_mae and cl_args.completion and j!= 1:
                continue
            data_n = {}
            for k ,v in data.items():
                if k != 'path':
                    data_n[k] = (v[j:j+1, None] if len(v.shape) < 4 else v[j:j+1]) if cl_args.completion else v.unsqueeze(0) 
                else:
                    data_n[k] = v
            model.set_input(data_n)
            if cl_args.completion:
                model.reconstruct()

            else:
                with torch.no_grad():
                    model.forward()
            fetched_data = fetch_reals(data_n, lidar_A, device, opt.model.norm_label)
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
            if (cl_args.completion and j == 1) or cl_args.compression:
                scores = calc_supervised_metrics(synth_depth, fetched_data['depth'], fetched_data['mask'], fetched_data['points'], lidar_A)
                for k, v in scores.items():
                    t_scores[k].append(v)
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
            if cl_args.completion:
                visualizer.display_current_results('',current_visuals, [seq, _id, cl_args.on_input, False, 'completion_' + str(j) if cl_args.completion else ''], ds_cfg, opt.dataset.dataset_A.name, lidar_A, ds_cfg_ref,\
                        cl_args.ref_dataset_name ,lidar, save_img=True)
            # else:
                # visualizer.display_current_results('', current_visuals, (seq, id),ds_cfg, opt.dataset.dataset_A.name, lidar, save_img=True)
        tq.update(1)
    if cl_args.completion or cl_args.compression:
        for k, v in t_scores.items():
            t_scores[k] = int(np.mean(v) * 1e4) / 1e4
        print(t_scores)


if __name__ == '__main__':
    main()
    
 