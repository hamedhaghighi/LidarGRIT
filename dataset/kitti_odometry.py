
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from glob import glob
import random
from util.lidar import point_cloud_to_xyz_image
from util import _map
from PIL import Image
from scipy import ndimage as nd
from collections import namedtuple
import torchvision.transforms as transforms


def car2hom(pc):
    return np.concatenate([pc[:, :3], np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=-1)


class  KITTIOdometry(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        DATA,
        shape=(64, 256),
        flip=False,
        modality=("depth"),
        fill_in_label=False,
        name='kitti', 
        limited_view=False,
        finesize=None,
        norm_label=False,
        is_ref_semposs=False,
        do_augment=False):
        super().__init__()
        self.root = osp.join(root, "sequences")
        if '/' in split:
            s1 , s2 = split.split('/')
            self.subsets = np.array(getattr(DATA.split, s1) + getattr(DATA.split, s2))
        else:
            self.subsets = np.asarray(getattr(DATA.split, split))
        self.shape = tuple(shape)
        self.min_depth = DATA.min_depth
        self.max_depth = DATA.max_depth
        self.flip = flip
        assert "depth" in modality, '"depth" is required'
        self.modality = modality
        self.datalist = None
        self.DATA = DATA
        self.is_sorted = DATA.is_sorted
        self.is_raw = DATA.is_raw
        self.fill_in_label = fill_in_label
        self.name = name
        self.has_rgb = 'rgb' in modality
        self.has_label = 'label' in modality
        self.limited_view = limited_view
        self.finesize = finesize
        self.norm_label = norm_label
        self.is_ref_semposs = is_ref_semposs
        # Compose the transforms
        self.transform_list = [
            transforms.ToTensor(),
            transforms.Resize(self.shape, TF.InterpolationMode.NEAREST) 
            ]
        
        self.transform_aug_list = [transforms.RandomRotation(degrees=45), transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5),\
                                             transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1))] if do_augment and split == 'train' else None
        if self.has_rgb:
            self.has_rgb = True
            calib = self.load_calib()
            self.velo_to_camera_rect =calib.T_cam2_velo
            self.cam_intrinsic = calib.P_rect_20
        self.load_datalist()

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}
        sequence_path = os.path.join(self.root, '00')
        # Load the calibration file
        calib_filepath = os.path.join(sequence_path, 'calib.txt')
        filedata = {}

        with open(calib_filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    filedata[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        calib = namedtuple('CalibData', data.keys())(*data.values())
        return calib

    
    def fill(self, data, invalid=None):
        if invalid is None: invalid = np.isnan(data)
        ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
        return data[tuple(ind)]

    def load_datalist(self):
        datalist = []
        for subset in self.subsets:
            subset_dir = osp.join(self.root, str(subset).zfill(2))
            sub_point_paths = sorted(glob(osp.join(subset_dir, "velodyne/*")))
            datalist += list(sub_point_paths)
        self.datalist = datalist
        if self.has_label:
            self.label_list = [d.replace('velodyne', 'labels').replace('bin' if self.is_raw else 'npy', 'label' if self.is_raw else 'png') for d in self.datalist]
        if self.has_rgb:
            self.rgb_list = [d.replace('velodyne', 'image_2').replace('bin' if self.is_raw else 'npy', 'png') for d in self.datalist]
        self.tag_list = [d.replace('velodyne', 'tag').replace('bin', 'tag') for d in self.datalist] if self.name == 'semanticPOSS' and self.is_raw else None

    def preprocess(self, out):
        out["depth"] = np.linalg.norm(out["points"], ord=2, axis=2)
        # fill in label
        if 'label' in out and self.fill_in_label:
          fill_in_mask = ~ (out["depth"] > 0.0)
          out['label'] = self.fill(out['label'], fill_in_mask)
        if self.name == 'carla':
            fill_in_mask = ~ (out["depth"] > 0.0)
            out['depth'] = self.fill(out['depth'], fill_in_mask)
            if 'reflectance' in out:
                out['reflectance'] = self.fill(out['reflectance'], fill_in_mask)
            if 'rgb' in out:
                out['rgb'] = self.fill(out['rgb'], fill_in_mask)
        ## create mask
        mask = (
            (out["depth"] > 0.0)
            & (out["depth"] > self.min_depth)
            & (out["depth"] < self.max_depth)
        )
        out["depth"] -= self.min_depth
        out["depth"] /= self.max_depth - self.min_depth
        out["mask"] = mask
        out["points"] /= self.max_depth  # unit space
        ## masking out modalities other than  label or rgb
        for key in out.keys():
            if (key == 'label' and self.fill_in_label) or key == 'rgb':
                continue
            if key == 'rgb':
                out[key][~np.repeat(mask[:, :, None], 3, 2)] = 0
            else:
                out[key][~mask] = 0
        return out

    def transform(self, out):
        concatenated_image = np.concatenate([image[..., None] if len(image.shape) < 3 else image for image in out.values()], axis=-1)
        for trfm in self.transform_list:
            concatenated_image = trfm(concatenated_image)
        
        if random.random() < 0.5 and self.transform_aug_list is not None:
            for trfm in self.transform_aug_list:
                concatenated_image = trfm(concatenated_image)
        ret = {}
        pre_idx = 0
        for key, image in out.items():
            num_channels = 1 if len(image.shape) < 3 else image.shape[-1]
            transformed_image = concatenated_image[pre_idx: pre_idx + num_channels]
            ret[key] = transformed_image
            pre_idx += num_channels
        
        # for k, v in out.items():
        #     v = TF.to_tensor(v)
        #     v = TF.resize(v, self.shape, TF.InterpolationMode.NEAREST)
        #     out[k] = v
        return ret

    def image_to_pcl(self, rgb_image, point_cloud):
        rgb = np.zeros((len(point_cloud),3), dtype=np.int32)
        height, width, _ = rgb_image.shape
        hom_pcl_points = car2hom(point_cloud[:, :3]).T
        pcl_in_cam_rect = np.dot(self.velo_to_camera_rect, hom_pcl_points)
        pcl_in_image = np.dot(self.cam_intrinsic, pcl_in_cam_rect)
        pcl_in_image = np.array([pcl_in_image[0] / pcl_in_image[2], pcl_in_image[1] / pcl_in_image[2], pcl_in_image[2]])
        canvas_mask = (pcl_in_image[0] > 0.0) & (pcl_in_image[0] < width) & (pcl_in_image[1] > 0.0)\
            & (pcl_in_image[1] < height) & (pcl_in_image[2] > 0.0)
        valid_pcl_in_image = pcl_in_image[:, canvas_mask].astype('int32')
        rgb[canvas_mask] = rgb_image[valid_pcl_in_image[1], valid_pcl_in_image[0], :]
        return rgb

    def __getitem__(self, index):
        points_path = self.datalist[index]
        if not self.is_raw:
            points = np.load(points_path).astype(np.float32)
            if self.has_label:
                labels_path = self.label_list[index]
                sem_label = np.array(Image.open(labels_path))
                if self.name == 'semanticPOSS':
                    sem_label = _map(sem_label, self.DATA.learning_map)
                sem_label = _map(sem_label, self.DATA.m_learning_map)
                points = np.concatenate([points, sem_label.astype('float32')[..., None]], axis=-1)
            if self.has_rgb:
                rgb_path = self.rgb_list[index]
                rgb = np.array(Image.open(rgb_path))
                points = np.concatenate([points, rgb.astype('float32')], axis=-1)
            # if self.limited_view:
            #     _, W, _ = points.shape
            #     points = points[:, int(3*W/8) : int(5*W/8), :]

        else:
            point_cloud = np.fromfile(points_path, dtype=np.float32).reshape((-1, 4))
            if self.has_label:
                labels_path = self.label_list[index]
                if self.name in ['kitti', 'carla', 'semanticPOSS']:
                    label = np.fromfile(labels_path, dtype=np.int32)
                    sem_label = label & 0xFFFF 
                    sem_label = _map(_map(sem_label, self.DATA.learning_map), self.DATA.m_learning_map)
                elif self.name == 'synthlidar':
                    sem_label = np.fromfile(labels_path, dtype=np.uint32)
                    sem_label = _map(_map(sem_label, self.DATA.learning_map), self.DATA.m_learning_map)
                point_cloud = np.concatenate([point_cloud, sem_label.astype('float32')[:, None]], axis=1)
            if self.has_rgb:
                rgb_path = self.rgb_list[index]
                rgb_image = np.array(Image.open(rgb_path))
                rgb = self.image_to_pcl(rgb_image, point_cloud)
                point_cloud = np.concatenate([point_cloud, rgb.astype('float32')], axis=1)
            H , W = self.DATA.height, self.DATA.width
            fov_up, fov_down = self.DATA.fov_up, self.DATA.fov_down
            
            tag = np.fromfile(self.tag_list[index], dtype = np.bool) if self.tag_list is not None else None
            points, _ = point_cloud_to_xyz_image(point_cloud, H=H, W=W, fov_down=fov_down, fov_up=fov_up, is_sorted=self.is_sorted, limited_view=self.limited_view, tag=tag)
            
        out = {}
        out["points"] = points[..., :3]
        if "reflectance" in self.modality:
            out["reflectance"] = points[..., [3]]/255.0 if self.name == 'semanticPOSS' else points[..., [3]]
        if "label" in self.modality:
            out["label"] = points[..., [4]] / (10.0 if self.name == 'semanticPOSS' else 19.0) if self.norm_label else points[..., [4]]
            # if self.name == 'carla' :
            #     out['label'] = _map(out['label'].astype('int'), self.DATA.kitti_to_POSS_map).astype(np.float32)
            out['lwo'] = points[..., [4]]
            if self.name == 'carla' and self.is_ref_semposs:
                out['lwo'] = _map(out['lwo'].astype('int'), self.DATA.kitti_to_POSS_map).astype(np.float32)
        if self.has_rgb:
            out["rgb"] = points[..., -3:]/ 255.0 # h,w, c
            black_pos = (out["rgb"] == 0.0).all(2)
            avg_color = np.mean(out["rgb"][~black_pos], axis=0)
            out["rgb"][black_pos] = avg_color
        out = self.preprocess(out)
        out = self.transform(out)
        out['path'] = points_path
        return out

    def __len__(self):
        return len(self.datalist)
