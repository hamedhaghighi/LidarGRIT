"""This package includes a miscellaneous collection of useful helper functions."""
import gc
import os
import re

import os.path as osp
from plyfile import PlyData
import subprocess

import cv2
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from util.geometry import estimate_surface_normal

m2ch = {'label':1, 'rgb':3, 'reflectance':1, 'mask':1, 'inv':1, 'depth':1}


def class_to_dict(obj):
    members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    dict_ = {}
    for k in members:
        v = getattr(obj, k)
        if not isinstance(v, (int, float, str, bool, list, tuple, dict, np.ndarray, torch.Tensor, type(None))):
            dict_[k] = class_to_dict(v)
        else:
            dict_[k] = v
    return dict_

def disentangle_output(output, out_ch, gumbel, out_modality, soft_mask=True):
    output_dict = {}
    i = 0
    for k in out_ch:
        output_dict[k] = output[:, i : i + m2ch[k]]
        i = i + m2ch[k]
    if 'mask' in output_dict:
        output_dict['mask_logit'] = output_dict['mask']
        if soft_mask:
            mask = output_dict['mask'] = gumbel(output_dict['mask_logit'])
        else:
            prob = torch.sigmoid(output_dict['mask_logit'].detach())
            mask = output_dict['mask'] = (torch.bernoulli(prob)).float()
        if 'depth' in output_dict:
            depth = torch.tanh(output_dict['depth'])
            output_dict['depth_orig'] = depth
            output_dict['depth'] = mask * depth + (1 - mask) * -1
        if 'reflectance' in output_dict:
            output_dict['reflectance_orig'] = output_dict['reflectance']
            r = torch.tanh(output_dict['reflectance'])
            output_dict['reflectance'] = mask * r + (1 - mask) * -1
    else:
        if 'depth' in output_dict:
            depth = torch.tanh(output_dict['depth'])
            output_dict['depth'] = depth
        if 'reflectance' in output_dict:
            r = torch.tanh(output_dict['reflectance'])
            output_dict['reflectance'] = r
        output_dict['mask'] = 1 - (output_dict['depth'] <= -0.9).float()
        output_dict['mask_logit'] = output_dict['mask']
    out_list = []
    for m in out_modality:
        out_list.append(output_dict[m])
    out = torch.cat(out_list, dim=1)
    return output_dict, out

def make_class_from_dict(opt):
    if any([isinstance(k, int) for k in opt.keys()]):
        return opt
    else:
        class dict_class():
            def __init__(self):
                for k , v in opt.items():
                    if isinstance(v , dict):
                        setattr(self, k, make_class_from_dict(v)) 
                    else:
                        setattr(self, k, v)
        return dict_class()
    
labels_mapping = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 1,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7,
    3: 7,
    4: 7,
    6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16
}
class SphericalOptimizer(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.params = params

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for param in self.params:
            param.data.div_(param.pow(2).mean(dim=1, keepdim=True).add(1e-9).sqrt())
        return loss
    



def prepare_data_for_seg(data, lidar, is_batch=True):
    depth = data['depth'] * (lidar.max_depth - lidar.min_depth) + lidar.min_depth
    points = data['points'] * lidar.max_depth
    vol = torch.cat([depth, points, data['reflectance'], data['mask']], dim=1 if is_batch else 0)
    return vol
 
def prepare_synth_for_seg(model, lidar, tag='synth'):
    if hasattr(model, tag + '_reflectance'):
        synth_reflectance = getattr(model, tag + '_reflectance')
    if hasattr(model, tag + '_mask'):
        synth_mask = getattr(model, tag + '_mask')
    if hasattr(model, tag + '_depth'):
        synth_depth = getattr(model, tag + '_depth')
    synth_depth = lidar.denormalize_depth(tanh_to_sigmoid(synth_depth))
    synth_points = lidar.depth_to_xyz(tanh_to_sigmoid(synth_depth)) * lidar.max_depth
    synth_reflectance = tanh_to_sigmoid(synth_reflectance)
    vol = torch.cat([synth_depth, synth_points, synth_reflectance, synth_mask], dim=1)
    return vol
 
def cat_modality(data_dict, modality):
    data_list = []
    for m in modality:
        assert m in data_dict
        data_list.append(data_dict[m])
    out = torch.cat(data_list, dim=1)
    return out
    
def fetch_reals(data, lidar, device, norm_label=False):
    mask = data["mask"].float()
    # depth = lidar.normalize_depth(data["depth"])
    depth = data["depth"]   # [0,1]
    depth = sigmoid_to_tanh(depth)  # [-1,1]
    depth = mask * depth + (1 - mask) * -1
    
    batch = {'mask': mask, 'depth': depth, 'points': data['points']}
    if 'lwo' in data:
        batch['lwo'] = data['lwo']
    if 'reflectance' in data:
        reflectance =  data["reflectance"] # [0, 1]
        reflectance = sigmoid_to_tanh(reflectance)
        reflectance = mask * reflectance + (1 - mask) * -1
        batch['reflectance'] = reflectance
    if 'rgb' in data:
        batch['rgb'] = sigmoid_to_tanh(data['rgb'])
    if 'label' in data: 
        batch['label'] = sigmoid_to_tanh(data['label']) if norm_label else data['label']
    for k , v in batch.items():
        batch[k] = v.to(device)
    batch['path'] = data['path']
    return batch


def init_weights(cfg):
    init_type = cfg.init.type
    gain = cfg.init.gain
    nonlinearity = cfg.relu_type

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif init_type == "kaiming":
                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(m.weight, 0, "fan_in", "relu")
                elif nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, 0.2, "fan_in", "learky_relu")
                else:
                    raise NotImplementedError(f"Unknown nonlinearity: {nonlinearity}")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f"Unknown initialization: {init_type}")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


def set_requires_grad(net, requires_grad: bool = True):
    for param in net.parameters():
        param.requires_grad = requires_grad


def zero_grad(optim):
    for group in optim.param_groups:
        for p in group["params"]:
            p.grad = None


def sigmoid_to_tanh(x: torch.Tensor):
    """[0,1] -> [-1,+1]"""
    out = x * 2.0 - 1.0
    return out


def tanh_to_sigmoid(x: torch.Tensor):
    """[-1,+1] -> [0,1]"""
    out = (x + 1.0) / 2.0
    return out


def get_device(cuda: bool):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        for i in range(torch.cuda.device_count()):
            print("device {}: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        print("device: CPU")
    return device


def noise(tensor: torch.Tensor, std: float = 0.1):
    noise = tensor.clone().normal_(0, std)
    return tensor + noise


def print_gc():
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def postprocess(synth, lidar, tol=1e-8, data_maps=None, dataset_name='kitti', norm_label=False):
    out = {}
    for key, value in synth.items():
        if 'depth' in key:
            out[key] = tanh_to_sigmoid(value).clamp_(0, 1)
            if not 'depth_orig' in key  and not 'aug' in key:
                out[key.replace('depth', 'points')] = lidar.depth_to_xyz(out[key], tol)
        elif "reflectance" in key:
            out[key] = tanh_to_sigmoid(value).clamp_(0, 1)
        elif 'label' in key:
            if dataset_name in ['kitti', 'carla', 'semanticPOSS']:
                if norm_label and key != 'synth_label':
                    value = tanh_to_sigmoid(value)
                    value = torch.round(value * (10.0 if  dataset_name == 'semanticPOSS' else 19.0))
                label_tensor = _map(_map(value.squeeze(dim=1).long(), data_maps.learning_map_inv), data_maps.color_map)
                out[key] = torch.flip(label_tensor.permute(0, 3, 1, 2), dims=(1,))
            elif dataset_name == 'nuscene':
                label_tensor = _map(_map(_map(value.squeeze().long(), labels_mapping), data_maps.learning_map_inv), data_maps.color_map)
                out[key] = torch.flip(label_tensor.permute(0, 3, 1, 2), dims=(1,))
        elif 'rgb' in key:
            out[key] = tanh_to_sigmoid(value).clamp_(0, 1) * 255.0
        else:
            out[key] = value
    return out


def save_videos(frames, filename, fps=30.0):
    N = len(frames)
    H, W, C = frames[0].shape
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename + ".mp4", codec, fps, (W, H))
    for frame in tqdm(frames, desc="Writing..."):
        writer.write(frame[..., ::-1])
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", filename)

def colorize(tensor, cmap="turbo", vmax=1.0):
    assert tensor.ndim == 2, "got {}".format(tensor.ndim)
    normalizer = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    tensor = mapper.to_rgba(tensor)[..., :3]
    return tensor



def flatten(tensor_BCHW):
    return tensor_BCHW.flatten(2).permute(0, 2, 1).contiguous()


def xyz_to_normal(xyz, mode="closest"):
    normals = -estimate_surface_normal(xyz, mode=mode)
    normals[normals != normals] = 0.0
    normals = tanh_to_sigmoid(normals).clamp_(0.0, 1.0)
    return normals


class SphericalOptimizer(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.params = params

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for param in self.params:
            param.data.div_(param.pow(2).mean(dim=1, keepdim=True).add(1e-9).sqrt())
        return loss


def masked_loss(img_ref, img_gen, mask, distance="l1"):
    if distance == "l1":
        loss = F.l1_loss(img_ref, img_gen, reduction="none")
    elif distance == "l2":
        loss = F.mse_loss(img_ref, img_gen, reduction="none")
    else:
        raise NotImplementedError
    loss = (loss * mask).sum(dim=(1, 2, 3))
    loss = loss / mask.sum(dim=(1, 2, 3))
    return loss


def _map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    if torch.is_tensor(label):
        lut = torch.from_numpy(lut).to(label.get_device()).long()
    return lut[label]

PCERRORPATH = "./pc_error"
TEMPPATH = "temp/data/"
def pcerror(pcRefer,pc,pcReferNorm,pcerror_cfg_params, pcerror_result,pcerror_path=PCERRORPATH):
  '''
  Options: 
          --help=0            This help text
    -a,   --fileA=""          Input file 1, original version
    -b,   --fileB=""          Input file 2, processed version
    -n,   --inputNorm=""      File name to import the normals of original point
                              cloud, if different from original file 1n
    -s,   --singlePass=0      Force running a single pass, where the loop is
                              over the original point cloud
    -d,   --hausdorff=0       Send the Haursdorff metric as well
    -c,   --color=0           Check color distortion as well
    -l,   --lidar=0           Check lidar reflectance as well
    -r,   --resolution=0      Specify the intrinsic resolution
          --dropdups=2        0(detect), 1(drop), 2(average) subsequent points
                              with same coordinates
          --neighborsProc=1   0(undefined), 1(average), 2(weighted average),
                              3(min), 4(max) neighbors with same geometric
                              distance
          --averageNormals=1  0(undefined), 1(average normal based on neighbors
                              with same geometric distance)
          --mseSpace=1        Colour space used for PSNR calculation
                              0: none (identity) 1: ITU-R BT.709 8: YCgCo-R
          --nbThreads=1       Number of threads used for parallel processing
  '''

  if pcerror_result is not None:
    pcLabel =os.path.basename(pcerror_result).split(".")[0]
  else:
    pcLabel = "pt0"
  if type(pc) is not str:
    write_ply_data(TEMPPATH + pcLabel + "pc.ply",pc)
    pc = TEMPPATH +pcLabel + "pc.ply"
  if type(pcRefer) is not str:
    write_ply_data(TEMPPATH+pcLabel+"pcRefer.ply",pcRefer)
    pcRefer = TEMPPATH + pcLabel + "pcRefer.ply"
  if pcerror_result is not None:
    f = open(pcerror_result, 'a+')
  else:
    import sys
    f = sys.stdout
  if type(pcerror_cfg_params) is str:
    pcerror_cfg_params = pcerror_cfg_params.split(' ')
  if pcReferNorm==None:
    process = subprocess.Popen([pcerror_path,
        '-a', pcRefer, '-b', pc] + pcerror_cfg_params,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  else:
    process = subprocess.Popen([pcerror_path, '-a', pcRefer, '-b', pc, '-n', pcReferNorm] + pcerror_cfg_params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()

  stdout = stdout.decode('utf-8')
  stderr = stderr.decode('utf-8')
  match = re.search(r'mse1,PSNR \(p2point\): ([\d.]+)', stdout)
  return float(match.group(1))
  

def write_ply_data(filename, points,attributeName=[],attriType=[]): 
    '''
    write data to ply file.
    e.g pt.write_ply_data('ScanNet_{:5d}.ply'.format(idx), np.hstack((point,np.expand_dims(label,1) )) , attributeName=['intensity'], attriType =['uint16'])
    '''
    # if os.path.exists(filename):
    #   os.system('rm '+filename)
    if type(points) is list:
      points = np.array(points)

    attrNum = len(attributeName)
    assert points.shape[1]>=(attrNum+3)

    if os.path.dirname(filename)!='' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename)) 

    plyheader = ['ply\n','format ascii 1.0\n'] + ['element vertex '+str(points.shape[0])+'\n'] + ['property float x\n','property float y\n','property float z\n']
    for aN,attrName in enumerate(attributeName):
      plyheader.extend(['property '+attriType[aN]+' '+ attrName+'\n'])
    plyheader.append('end_header')
    typeList = {'uint16':"%d",'float':"float",'uchar':"%d"}

    np.savetxt(filename, points, newline="\n",fmt=["%f","%f","%f"]+[typeList[t] for t in attriType],header=''.join(plyheader),comments='')

    return