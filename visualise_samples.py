# from data import create_dataset
import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from util import *
from util.lidar import LiDAR
from util.sampling.fps import downsample_point_clouds
from util.visualizer import visualize_tensor

os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def depth_to_xyz(depth, lidar, tol=1e-8, cpu=False):
    depth = tanh_to_sigmoid(depth).clamp_(0, 1)
    xyz = lidar.depth_to_xyz(depth, tol)
    xyz_out = xyz.flatten(2)
    xyz = xyz_out.transpose(1, 2)  # (B,N,3)
    if cpu:
        xyz = downsample_point_clouds(xyz.cuda(), 512).cpu()
    else:
        xyz = downsample_point_clouds(xyz, 512)
    return xyz, xyz_out

def to_np(tensor):
    return tensor.detach().cpu().numpy()

class Features10k(torch.utils.data.Dataset):
    def __init__(self, root, min_depth, max_depth, lidargen=False):
        self.lidargen = lidargen
        self.min_depth, self.max_depth = min_depth, max_depth
        self.sample_path_list = sorted(Path(root).glob("*.pth"))
    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]
        img = torch.load(sample_path, map_location="cpu")
        if self.lidargen:
            assert img.shape[0] == 2, img.shape

        if self.lidargen:
            img[[0]] = torch.exp2(img[[0]]*6) - 1
        depth = img[[0]]
        mask = torch.logical_and(depth > self.min_depth, depth < self.max_depth).float()
        img = img * mask
        return img.float(), mask.float()

    def __len__(self):
        return len(self.sample_path_list)



def main(runner_cfg_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path of the config file')
    parser.add_argument('--sample_dir', type=str, default='', help='Path of the sample dir')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the sample dir')
    parser.add_argument('--ref_dataset_name', type=str, default='kitti_360', help='Path of the sample dir')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--lidargen', action='store_true', help='fast test of experiment')
    parser.add_argument('--on_real', action='store_true', help='fast test of experiment')
    parser.add_argument('--depth', action='store_true', help='fast test of experiment')

    cl_args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    min_depth, max_depth = 0.9 if cl_args.ref_dataset_name == 'kitti' else 1.45, 120.0 if cl_args.ref_dataset_name == 'kitti' else 80.0
    device = torch.device('cpu')
    data_dict = defaultdict(list)
    data_dict = torch.load(cl_args.data_dir, map_location=device)
    print('data_dict loaded ...')
    ds_cfg = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{cl_args.ref_dataset_name}_cfg.yml', 'r')))
    ds_cfg.min_depth, ds_cfg.max_depth = min_depth, max_depth
    #### Train & Validation Loop
    # Train loop
    data_dict['synth-2d'] = [] 
    data_dict['synth-3d'] = []
    # fid_samples = [] if fid_cls is not None else None
    h = 64 
    w = 1024 if cl_args.ref_dataset_name == 'kitti_360' else 256
    lidar = LiDAR(cfg=ds_cfg, height=h, width=w).to(device)
    sample_folder = cl_args.sample_dir.split(os.path.sep)[-1]
    if sample_folder == '': 
        sample_folder = cl_args.sample_dir.split(os.path.sep)[-2]
    if cl_args.on_real:
        save_dir = cl_args.sample_dir.replace(sample_folder, 'real_visualised_depth' if cl_args.depth else 'real_visualised_xyz')
    else:
        save_dir = cl_args.sample_dir.replace(sample_folder, f'{sample_folder}_visualised_depth' if cl_args.depth else f'{sample_folder}_visualised_xyz')

    os.makedirs(save_dir, exist_ok=True)
    dataset = Features10k(cl_args.sample_dir, min_depth, max_depth, cl_args.lidargen)
    gen_loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True, drop_last=False)
    N = 512 if cl_args.fast_test else  min(len(data_dict['real-2d']) * 8, 5000, len(dataset))
    gen_loader_iter = iter(gen_loader)
    n_gen_batch = N//8 if cl_args.fast_test else  len(gen_loader) 
    gen_tq = tqdm.tqdm(total=n_gen_batch, desc='Iterating gen data', position=0, leave=True)
    index = 0
    if cl_args.on_real:
        for j in tqdm.tqdm(range(n_gen_batch)):
            real_depth = data_dict['real-2d'][j]
            if cl_args.depth:
                for i in range(len(real_depth)):
                    plt.imsave(osp.join(save_dir, f'depth_{index}.png'), ((np.asarray((real_depth[i] + 1)/2.0)[0])*2.5).clip(0.0, 1.0), cmap='turbo')
                    index += 1
            else:
                _ , xyz_norm = depth_to_xyz(real_depth, lidar, cpu=True)
                for i in range(len(xyz_norm)):
                    _, pts_img = visualize_tensor(to_np(xyz_norm.transpose(1, 2)[i]), (to_np((real_depth[i] + 1)/2)*2.5).clip(0.0, 1.0), 'point', cl_args.ref_dataset_name)
                    plt.imsave(osp.join(save_dir, f'point_{index}.png'), np.asarray(pts_img))
                    index += 1
    else:
            
        for _ in range(n_gen_batch):
            img, mask = next(gen_loader_iter)
            img = img.to(device)
            mask = mask.to(device)
            synth_depth = (img[:, [0]] - min_depth) / (max_depth - min_depth)
            synth_depth = synth_depth * mask

            if cl_args.depth:

                for i in range(len(synth_depth)):
                    plt.imsave(osp.join(save_dir, f'depth_{index}.png'), (np.asarray((synth_depth[i][0]))*2.5).clip(0.0, 1.0), cmap='turbo')
                    index += 1
            else:   
                synth_depth = synth_depth * 2 - 1
                _ , xyz_norm = depth_to_xyz(synth_depth, lidar, cpu=True)
                for i in range(len(xyz_norm)):
                    _, pts_img = visualize_tensor(to_np(xyz_norm.transpose(1, 2)[i]), (to_np((synth_depth[i] + 1)/2)*2.5).clip(0.0, 1.0), 'point', cl_args.ref_dataset_name)
                    plt.imsave(osp.join(save_dir, f'point_{index}.png'), np.asarray(pts_img))
                    index += 1
    
            gen_tq.update(1)
    ##### calculating unsupervised metrics


if __name__ == '__main__':
    main()
    
 