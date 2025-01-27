import argparse
import hashlib
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from dataset.datahandler import get_data_loader
# from data import create_dataset
from models import create_model
from rangenet.tasks.semantic.modules.segmentator import *
from util import *
from util.lidar import LiDAR
from util.metrics import bev
from util.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from util.metrics.fid import FID
from util.metrics.fpd import FPD
from util.metrics.jsd import compute_jsd
from util.metrics.seg_accuracy import compute_seg_accuracy
from util.metrics.swd import compute_swd
from util.sampling.fps import downsample_point_clouds
from util.visualizer import Visualizer, visualize_tensor

os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def depth_to_xyz(depth, lidar, tol=1e-8, cpu=False, num_points=512):
    depth = tanh_to_sigmoid(depth).clamp_(0, 1)
    xyz = lidar.depth_to_xyz(depth, tol)
    xyz_out = xyz.flatten(2)
    xyz = xyz_out.transpose(1, 2)  # (B,N,3)
    if cpu:
        xyz = downsample_point_clouds(xyz.cuda(), num_points).cpu()
    else:
        xyz = downsample_point_clouds(xyz, num_points)
    return xyz, xyz_out

def to_np(tensor):
    return tensor.detach().cpu().numpy()

class Features10k(torch.utils.data.Dataset):
    def __init__(self, root, min_depth, max_depth, lidargen=False, fpd=False, ref_dataset_name='kitti'):
        self.lidargen = lidargen
        self.fpd = fpd
        self.sample_path_list = sorted(Path(root).glob("*.pth"))
        self.min_depth, self.max_depth = min_depth, max_depth
        self.ref_dataset_name = ref_dataset_name
    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]
        img = torch.load(sample_path, map_location="cpu")
        if self.lidargen:
            assert img.shape[0] == 2, img.shape

        if self.lidargen:
            img[[0]] = torch.exp2(img[[0]]*6) - 1
        depth = img[[0]]
        if self.fpd and self.ref_dataset_name == 'kitti_360':
            mask = torch.logical_and(depth > 0.5, depth < 63.0).float() # based on lidargen evaluation
        else:
            mask = torch.logical_and(depth > self.min_depth, depth < self.max_depth).float()
        img = img * mask
        return img.float(), mask.float()

    def __len__(self):
        return len(self.sample_path_list)


def subsample(batch, n):
    if len(batch) <= n:
        return batch
    else:
        return batch[torch.linspace(0, len(batch), n + 1)[:-1].long()]



def main(runner_cfg_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, default='', help='Path of the sample dir')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the sample dir')
    parser.add_argument('--ref_dataset_name', type=str, default='kitti_360', help='Path of the sample dir')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--cpu', action='store_true', help='run on cpu')
    parser.add_argument('--fpd', action='store_true', help='calculate fpd or other metrics (fpd requires different pre-processing)')
    parser.add_argument('--debug', action='store_true', help='debugging through visualisation of samples')
    parser.add_argument('--lidargen', action='store_true', help='set true if you want to evaluate lidargen')
    parser.add_argument('--gpu', type=int, default=0, help='GPU no')
    parser.add_argument('--num_points', type=int, default=512, help='num of points for fps')
    parser.add_argument('--num_samples', type=int, default=5000, help='num of samples for comparison')
    cl_args = parser.parse_args()
    EVAL_MAX_DEPTH = 80.0
    EVAL_MIN_DEPTH = 1.45
    if cl_args.ref_dataset_name == 'kitti':
        EVAL_MAX_DEPTH = 120.0
        EVAL_MIN_DEPTH = 0.9
    if not cl_args.cpu:
        torch.cuda.set_device(f'cuda:{cl_args.gpu}')
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device(f'cuda:{cl_args.gpu}') if not cl_args.cpu else torch.device('cpu')
    data_dict = defaultdict(list)
    data_dict = torch.load(cl_args.data_dir, map_location=device)
    print('data_dict loaded ...')
    ds_cfg = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{cl_args.ref_dataset_name}_cfg.yml', 'r')))
    ds_cfg.min_depth, ds_cfg.max_depth = EVAL_MIN_DEPTH, EVAL_MAX_DEPTH
    min_depth , max_depth = ds_cfg.min_depth, ds_cfg.max_depth
    fpd_cls = FPD(None, cl_args.ref_dataset_name, None, max_sample=cl_args.num_samples, device='cuda')
    #### Train & Validation Loop
    # Train loop
    data_dict['synth-2d'] = [] 
    data_dict['synth-3d'] = []
    if cl_args.ref_dataset_name == 'kitti_360'and not cl_args.fpd:
        data_dict['synth-bev'] = []
    fpd_points = []
    h = 64 
    w = 1024 if cl_args.ref_dataset_name == 'kitti_360' else 256
    lidar = LiDAR(cfg=ds_cfg, height=h, width=w).to(device)
    dataset = Features10k(cl_args.sample_dir, min_depth, max_depth, cl_args.lidargen, cl_args.fpd, cl_args.ref_dataset_name)
    gen_loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True, drop_last=False)
    N = 8 if cl_args.fast_test else  cl_args.num_samples
    gen_loader_iter = iter(gen_loader)
    n_gen_batch = N//8 if cl_args.fast_test else  len(gen_loader) 
    gen_tq = tqdm.tqdm(total=n_gen_batch, desc='Iterating gen data', position=0, leave=True)
    for i in range(n_gen_batch):
        img, mask = next(gen_loader_iter)
        img = img.to(device)
        mask = mask.to(device)
        synth_depth = (img[:, [0]] - min_depth) / (max_depth - min_depth)
        synth_depth = synth_depth * mask
        synth_depth = synth_depth * 2 - 1
        data_dict['synth-2d'].append(synth_depth)
        xyz_ds , xyz_norm = depth_to_xyz(synth_depth, lidar, cpu=cl_args.cpu, num_points=cl_args.num_points)
        if cl_args.debug:
            import matplotlib.pyplot as plt
            plt.figure(0)
            plt.imshow(synth_depth[0,0].numpy(), cmap='jet')
            plt.figure(1)
            plt.imshow(data_dict['real-2d'][0][0,0].cpu().numpy(), cmap='jet')
            plt.figure(2)
            bev_img, pts_img = visualize_tensor(to_np(xyz_norm.transpose(1, 2)[0]), to_np((synth_depth[0] + 1)/2), 'point', cl_args.ref_dataset_name)
            plt.imshow(np.asarray(pts_img))
            plt.figure(3)
            bev_img, pts_img = visualize_tensor(to_np(((img[:, 1:4]).flatten(2).transpose(1, 2) / max_depth)[0]),\
             to_np((synth_depth[0] + 1)/2), 'point', cl_args.ref_dataset_name)
            plt.imshow(np.asarray(pts_img))
            plt.show()
            exit(1)
        data_dict['synth-3d'].append(xyz_ds)
        if cl_args.ref_dataset_name == 'kitti_360' and not cl_args.fpd:
            for pc in xyz_norm.transpose(1, 2):
                hist = bev.point_cloud_to_histogram(pc * max_depth)
                data_dict['synth-bev'].append(hist[None, ...].to(device))
        fpd_points.append(xyz_norm)
        gen_tq.update(1)
    ##### calculating unsupervised metrics
    for k ,v in data_dict.items():
        if isinstance(v, list):
            data_dict[k] = torch.cat(v, dim=0)[: N]
    n_subsample = 5000 if cl_args.ref_dataset_name == 'kitti' else 5000
    if cl_args.fpd:
        fpd_score = fpd_cls.fpd_score(torch.cat(fpd_points, dim=0))
        print(f'fpd: {fpd_score}')
    else:
        scores = {}
        scores.update(compute_swd(subsample(data_dict["synth-2d"], n_subsample), subsample(data_dict["real-2d"], n_subsample)))
        scores["jsd"] = compute_jsd(subsample(data_dict["synth-3d"], n_subsample) / 2.0, subsample(data_dict["real-3d"], n_subsample) / 2.0)
        scores.update(compute_cov_mmd_1nna(subsample(data_dict["synth-3d"], n_subsample), subsample(data_dict["real-3d"], n_subsample), 512, ("cd",)))
        if cl_args.ref_dataset_name == 'kitti_360':
                scores["jsd-bev"] = bev.compute_jsd_2d(data_dict["real-bev"], data_dict["synth-bev"])
                scores["mmd-bev"] = bev.compute_mmd_2d(data_dict["real-bev"], data_dict["synth-bev"])
        print(scores)


if __name__ == '__main__':
    main()
    
 