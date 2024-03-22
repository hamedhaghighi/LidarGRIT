import time
# from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.metrics.fid import FID
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
from util.metrics.fpd import FPD 
from util.metrics import bev  
import random
import hashlib
from pathlib import Path
from torch.utils.data import DataLoader
from util.visualizer import visualize_tensor

os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EVAL_MAX_DEPTH = 63.0
EVAL_MIN_DEPTH = 0.5
DATASET_MAX_DEPTH = 80.0

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
    def __init__(self, root, N, lidargen=False):
        self.lidargen = lidargen
        self.sample_path_list = sorted(Path(root).glob("*.pth"))[:N]
    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]
        img = torch.load(sample_path, map_location="cpu")
        if self.lidargen:
            assert img.shape[0] == 2, img.shape
        else:
            assert img.shape[0] == 5, img.shape

        if self.lidargen:
            img[[0]] = torch.exp2(img[[0]]*6) - 1
        depth = img[[0]]
        mask = torch.logical_and(depth > EVAL_MIN_DEPTH, depth < EVAL_MAX_DEPTH).float()
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
    parser.add_argument('--cfg', type=str, default='', help='Path of the config file')
    parser.add_argument('--sample_dir', type=str, default='', help='Path of the sample dir')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the sample dir')
    parser.add_argument('--ref_dataset_name', type=str, default='kitti_360', help='Path of the sample dir')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--cpu', action='store_true', help='fast test of experiment')
    parser.add_argument('--lidargen', action='store_true', help='fast test of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='GPU no')

    cl_args = parser.parse_args()
    if not cl_args.cpu:
        torch.cuda.set_device(f'cuda:{cl_args.gpu}')
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
  # create a visualizer that display/save images and plots
    ## initilisation of the model for netF in cut
    device = torch.device(f'cuda:{cl_args.gpu}') if not cl_args.cpu else torch.device('cpu')
    data_dict = defaultdict(list)
    data_dict = torch.load(cl_args.data_dir, map_location=device)
    N = 1 * 8 if cl_args.fast_test else  min(len(data_dict['real-2d']) * 8, 5000)
    print('data_dict loaded ...')
    ds_cfg = make_class_from_dict(yaml.safe_load(open(f'configs/dataset_cfg/{cl_args.ref_dataset_name}_cfg.yml', 'r')))
    min_depth , max_depth = ds_cfg.min_depth, ds_cfg.max_depth
    fpd_cls = FPD(None, cl_args.ref_dataset_name, None, device='cpu' if cl_args.cpu else 'cuda')
    #### Train & Validation Loop
    # Train loop
    data_dict['synth-2d'] = [] 
    data_dict['synth-3d'] = []
    if cl_args.ref_dataset_name == 'kitti_360':
        data_dict['synth-bev'] = []
    fpd_points = []
    # fid_samples = [] if fid_cls is not None else None
    h = 64 
    w = 1024 if cl_args.ref_dataset_name == 'kitti_360' else 256
    lidar = LiDAR(
    cfg=ds_cfg,
    height=h,
    width=w).to(device)
    dataset = Features10k(cl_args.sample_dir, N, cl_args.lidargen)
    gen_loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
    gen_loader_iter = iter(gen_loader)
    n_gen_batch = 1 if cl_args.fast_test else  len(gen_loader) 
    gen_tq = tqdm.tqdm(total=n_gen_batch, desc='Iterating gen data', position=0, leave=True)
    for i in range(n_gen_batch):
        img, _ = next(gen_loader_iter)
        img = img.to(device)
        synth_depth = (img[:, [0]] - min_depth) / (max_depth - min_depth)
        synth_depth = synth_depth * 2 - 1
        data_dict['synth-2d'].append(synth_depth)
        xyz_ds , xyz_norm = depth_to_xyz(synth_depth, lidar, cpu=cl_args.cpu)
        # import matplotlib.pyplot as plt
        # plt.figure(0)
        # plt.imshow(synth_depth[0,0].numpy(), cmap='jet')
        # plt.figure(1)
        # plt.imshow(data_dict['real-2d'][0][0,0].cpu().numpy(), cmap='jet')
        # plt.figure(2)
        # bev_img, pts_img = visualize_tensor(to_np(xyz_norm.transpose(1, 2)[0]), to_np((synth_depth[0] + 1)/2), 'point', cl_args.ref_dataset_name)
        # plt.imshow(np.asarray(pts_img))
        # plt.figure(3)
        # bev_img, pts_img = visualize_tensor(to_np(((img[:, 1:4]).flatten(2).transpose(1, 2) / max_depth)[0]), to_np((synth_depth[0] + 1)/2), 'point', cl_args.ref_dataset_name)
        # plt.imshow(np.asarray(pts_img))
        # plt.show()
        # exit(1)
        # xyz = (img[:, 1:4]).flatten(2).transpose(1, 2)
        # xyz_norm = xyz / max_depth
        # xyz_ds = downsample_point_clouds(xyz_norm, 512)
        data_dict['synth-3d'].append(xyz_ds)
        if cl_args.ref_dataset_name == 'kitti_360':
            for pc in xyz_norm.transpose(1, 2):
                hist = bev.point_cloud_to_histogram(pc * max_depth)
                data_dict['synth-bev'].append(hist[None, ...].to(device))
        fpd_points.append(xyz_norm)
        gen_tq.update(1)
    ##### calculating unsupervised metrics
    for k ,v in data_dict.items():
        if isinstance(v, list):
            data_dict[k] = torch.cat(v, dim=0)[: N]
    scores = {}
    scores.update(compute_swd(subsample(data_dict["synth-2d"], 2048), subsample(data_dict["real-2d"], 2048)))
    scores["jsd"] = compute_jsd(subsample(data_dict["synth-3d"], 2048) / 2.0, subsample(data_dict["real-3d"], 2048) / 2.0)
    scores.update(compute_cov_mmd_1nna(subsample(data_dict["synth-3d"], 2048), subsample(data_dict["real-3d"], 2048), 512, ("cd",)))
    if cl_args.ref_dataset_name == 'kitti_360':
            scores["jsd-bev"] = bev.compute_jsd_2d(data_dict["real-bev"], data_dict["synth-bev"])
            scores["mmd-bev"] = bev.compute_mmd_2d(data_dict["real-bev"], data_dict["synth-bev"])
    if fpd_cls is not None:
        scores['fpd'] = fpd_cls.fpd_score(torch.cat(fpd_points, dim=0))
    print(scores)


if __name__ == '__main__':
    main()
    
 