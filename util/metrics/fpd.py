#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import pickle

import numpy as np
import scipy
import torch
from tqdm import tqdm, trange

from util.metrics.pointnet import pretrained_pointnet


class FPD():
  def __init__(self, train_dataset, dataset_name, lidar, max_sample=5000, batch_size=8, device='cuda'):
    self.path = './'
    self.batch_size = batch_size
    ds = train_dataset
    n_samples = min(max_sample, len(train_dataset)) if train_dataset is not None else max_sample
    stat_root = os.path.join('stats', 'fpd_stats')
    os.makedirs(stat_root, exist_ok=True)
    stat_dir = os.path.join(stat_root, f'fpd_{dataset_name}.pkl') if n_samples == 5000 else os.path.join(stat_root, f'fpd_{dataset_name}_{10000}.pkl')
    # parameters
    self.lidar = lidar 
    # concatenate the encoder and the head
    self.device = torch.device(device)
    self.pointnet = pretrained_pointnet().to(self.device)

    if os.path.isfile(stat_dir):
        stat = pickle.load(open(stat_dir, 'rb'))
        self.mu_train, self.sigma_train = stat['mu'], stat['sigma']
        print('FPD stats loaded ...\n')
    else:
        sample_indxs = np.random.choice(range(len(train_dataset)), n_samples, replace=False)
        samples = []
        for ind in tqdm(sample_indxs, desc='gathering real samples for fpd'):
            data = ds[ind]
            if 'B' in data:
              data = data['B']
            depth = data['depth'].to(self.device)
            if dataset_name == 'kitti_360':
              mask_1 = data['mask'].to(self.device)
              unorm_depth = depth * (self.lidar.max_depth - self.lidar.min_depth) + self.lidar.min_depth
              mask_2 = torch.logical_and(unorm_depth > 0.5, unorm_depth < 63.0).float() # based on lidargen evaluation
              depth *= mask_1 * mask_2
            xyz = self.lidar.depth_to_xyz(depth[None, ...], 1e-8)[0]
            xyz = xyz.flatten(2)
            samples.append(xyz)
        samples = torch.stack(samples, dim=0).flatten(2)
        self.mu_train , self.sigma_train = self.compute_stats(samples)
        pickle.dump({'mu': self.mu_train, 'sigma':self.sigma_train}, open(stat_dir, 'wb'))
        print('FPD stats saved ...\n')

  def compute_stats(self, data_tensor):
    feature_array = self.compute_features(data_tensor)
    mu = np.mean(feature_array, axis=0)
    sigma = np.cov(feature_array, rowvar=False)
    return mu, sigma

  def compute_features(self, data_tensor):
    n_batch = np.ceil(len(data_tensor) / self.batch_size)
    features_list = []
    for i in trange(int(n_batch), desc='extracting features for fpd'):
      data = data_tensor[i * self.batch_size: (i + 1) * self.batch_size]
      if data.is_cuda:
        feature = self.pointnet(data)
      else:
        feature = self.pointnet(data.cuda())
      features_list.append(feature.detach().cpu().numpy())
    
    return np.concatenate(features_list, axis=0)

  
  def fpd_score(self, samples):
        # list of tensors in cpu
        assert samples.shape[0] > 1 , 'for FpD num of samples must be greater than one'
        # batch_size = min(batch_size, samples.shape[0])
        mu , sigma = self.compute_stats(samples)
        m = np.square(self.mu_train - mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(self.sigma_train, sigma), disp=False)
        distance = np.real(m + np.trace(self.sigma_train + sigma - s * 2))
        return float(distance)