import numpy as np
import os
import sys
import ntpath
import time
import datetime
import shutil
from . import util, html
from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from util import colorize, postprocess, flatten
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from glob import glob
import yaml
from PIL import Image
import wandb
from util import class_to_dict

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def visualize_tensor(pts, depth, tag, ds_name):

    color = plt.cm.jet(np.clip(depth, 0, 1).flatten())
    # color = depth
    # mask out invalid points
    xyz = pts
    color = color[..., :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd],zoom=0.25, front=[0.0, 0.0, 1.0],
    #                               lookat=[0.0, 0.0, 0.0],
    #                               up=[1.0, 0.0, 0.0])
    # offscreen rendering
    render = rendering.OffscreenRenderer(1920, 1080, headless=True)
    mtl = rendering.MaterialRecord()
    mtl.base_color = [1, 1, 1, 0.5]
    if ds_name == 'kitti' or ds_name == 'carla':
        mtl.point_size = 4 if 'synth' in tag else 4
    else:
        mtl.point_size = 4
    mtl.shader = "defaultLit"
    render.scene.add_geometry("point cloud", pcd, mtl)
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
    bev_img = render.render_to_image()
    render.setup_camera(60.0, [0, 0, 0], [-0.1, -0.2, 0.25], [0, 0, 1])
    pts_img = render.render_to_image()
    return bev_img, pts_img

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt.training
        self.exp_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)

        self.tb_dir = os.path.join(self.exp_dir +('/TB/' if self.opt.isTrain else '/TB_test/'), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.log_name = os.path.join(self.tb_dir , 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        self.writer.add_text('config', yaml.dump(class_to_dict(opt)).replace('\n', '  \n'))
        self.norm_label = opt.model.norm_label
        # wandb.init(project='vqgan', config=class_to_dict(opt))

    def log_imgs(self, tensor, tag, step, color=True, cmap='turbo', save_img=False, ds_name = 'carla'):
        B = tensor.shape[0]
        nrow = 4 if B > 8 else 1
        grid = make_grid(tensor.detach(), nrow=nrow)
        grid = grid.cpu().numpy()  # CHW
        if color:
            grid = grid[0]  # HW
            if cmap != 'gray' and 'depth' in tag:
                grid *= 2 if ds_name == 'carla' or ds_name == 'kitti' else 2
            if ds_name == 'semanticPOSS' and 'reflectance' in tag:
                grid *=5
            grid = colorize(grid, cmap=cmap).transpose(2, 0, 1)  # CHW
        else:
            grid = grid.astype(np.uint8)
        if save_img:
            if grid.max() <= 1.0:
                grid = grid * 255.0
            im_grid = Image.fromarray(grid.transpose(1,2,0).astype(np.uint8))
            folder_name = 'seq_' + str(step[0]).zfill(2)\
                  + '_id_' + str(step[1]).zfill(6) +\
                      ('_on_input' if step[2] else '') + ('_real' if step[3] else '')  + (step[4]) 
            img_folder_dir = os.path.join(self.exp_dir,'TB_test', 'img_results', folder_name)
            os.makedirs(img_folder_dir, exist_ok=True)
            im_grid.save(os.path.join(img_folder_dir, tag.replace('/', '_') + '.png'))
        else:
            self.writer.add_image(tag, grid, step)
            # wandb.log({tag: [wandb.Image(grid, caption=tag)]}, step=step)

    def display_current_results(self, phase, current_visuals, g_step, data_maps_A,\
         dataset_name_A,lidar_A, data_maps_B=None, dataset_name_B=None, lidar_B=None, save_img=False):
        def display_domain_visuals(visuals, g_step, data_maps, lidar, dataset_name):
            visuals = postprocess(visuals, lidar, data_maps=data_maps, dataset_name=dataset_name, norm_label=self.norm_label)
            for k , v in visuals.items():
                if 'points' in k:
                    points = flatten(v)
                    # inv = visuals['real_label' if k == 'real_points' else 'synth_label']
                    depth = visuals[k.replace('points', 'depth')]
                    # r = 1 / (inv + 0.1)
                    # r = (r - r.min()) / (r.max()-r.min()) 
                    image_list = []
                    for i in range(points.shape[0]):
                        _, gen_pts_img = visualize_tensor(to_np(points[i]), to_np(depth[i]) * 2, k, dataset_name)
                        # _, gen_pts_img = visualize_tensor(to_np(points[i]), to_np((inv[i].permute(1,2,0).flatten(0, 1))/255.0))
                        image_list.append(torch.from_numpy(np.asarray(gen_pts_img)))
                    visuals[k] = torch.stack(image_list, dim=0).permute(0, 3, 1, 2)
            for k , img_tensor in visuals.items():
                colorise = False if  any([k_ in k for k_ in ['points', 'label', 'rgb']]) else True
                cmap = 'viridis' if ('reflectance' in k) else ('gray' if 'mask' in k and not 'logit' in k else 'turbo')
                self.log_imgs(img_tensor, phase + '/' + k, g_step, colorise, cmap, save_img, dataset_name)
        if lidar_B is not None:
            domain_A_keys = [k for k in list(current_visuals.keys()) if not 'B' in k and not 'synth' in k]
            domain_B_keys = [k for k in list(current_visuals.keys()) if k not in domain_A_keys]
            visual_A, visual_B = {k:current_visuals[k] for k in domain_A_keys}, {k:current_visuals[k] for k in domain_B_keys}
            display_domain_visuals(visual_A, g_step, data_maps_A, lidar_A, dataset_name_A)
            display_domain_visuals(visual_B, g_step, data_maps_B, lidar_B, dataset_name_B)
        else:
            display_domain_visuals(current_visuals, g_step, data_maps_A, lidar_A, dataset_name_A)

    def plot_current_losses(self, phase, epoch, losses, g_step):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for tag , loss in losses.items():
            self.writer.add_scalar(phase + '/' + tag, loss, g_step)
            # wandb.log({phase + '/' + tag: loss}, step=g_step)

        # plotting epoch    
        self.writer.add_scalar('epoch', epoch, g_step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, phase,epoch, iters, losses, tq=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = 'Validation\n' if phase == 'val' else ''
        message = message + '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.8f ' % (k, v)
        if not tq:
            print(message)  # print the message
        else:
            tq.write(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
