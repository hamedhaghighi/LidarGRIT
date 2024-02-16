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
import random


os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def depth_to_xyz(depth, lidar, tol=1e-8):
    depth = tanh_to_sigmoid(depth).clamp_(0, 1)
    xyz = lidar.depth_to_xyz(depth, tol)
    xyz_out = xyz.flatten(2)
    xyz = xyz_out.transpose(1, 2)  # (B,N,3)
    xyz = downsample_point_clouds(xyz, 512)
    return xyz, xyz_out


class M_parser():
    def __init__(self, cfg_path, data_dir, data_dir_B, load, is_test):
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
        opt_t.name = 'test'
    else:
        if 'pix2pix' in opt_m.name:
            opt_t.name = f'pix2pix_modality_A_{modality_A}_out_ch_{out_ch}_L_L1_{opt_m.lambda_L1}_L_nd_{opt_m.lambda_nd}' \
                + f'_L_GAN_{opt_m.lambda_LGAN}_L_mask_{opt_m.lambda_mask}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}'
        elif 'vqgan' in opt_m.name:
            losscfg = opt_m.vqmodel.lossconfig.params
            opt_t.name = f'vqgan_modality_A_{modality_A}_out_ch_{out_ch}_L_nd_{losscfg.lambda_nd}_L_disc_{losscfg.disc_weight}' \
                + f'_d_start_{losscfg.disc_start}_L_mask_{losscfg.lambda_mask}_w_{opt_d.img_prop.width}_h_{opt_d.img_prop.height}_bs_{opt_t.batch_size}'
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
    parser.add_argument('--load', type=str, default='', help='the name of the experiment folder while loading the experiment')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--fast_test', action='store_true', help='fast test of experiment')
    parser.add_argument('--ref_dataset_name', type=str, default='', help='reference dataset name for measuring unsupervised metrics')
    parser.add_argument('--n_fid', type=int, default=1000, help='num of samples for calculation of fid')
    parser.add_argument('--no_inv', action='store_true', help='use it to calc unsupervised metrics on input inv, in case modality_B does not contain inv')
    parser.add_argument('--on_input', action='store_true', help='unsupervised metrics will be calculated on dataset A')
    parser.add_argument('--gpu', type=int, default=0, help='GPU no')

    cl_args = parser.parse_args()
    torch.cuda.set_device(f'cuda:{cl_args.gpu}')
    if runner_cfg_path is not None:
        cl_args.cfg = runner_cfg_path
    if 'checkpoints' in cl_args.cfg:
        cl_args.load = cl_args.cfg.split(os.path.sep)[1]
    opt = M_parser(cl_args.cfg, cl_args.data_dir, cl_args.data_dir_B, cl_args.load, cl_args.test)
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
    is_transformer = 'transformer' in opt.model.name
    opt.training.gpu_ids = [cl_args.gpu]
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
    min_rmse = 10000
    if cl_args.ref_dataset_name == 'kitti':
        ignore_label = [0, 2, 3, 4, 6, 5, 7, 8, 10, 12, 16]
    elif cl_args.ref_dataset_name == 'semanticPOSS':
        ignore_label = [0, 3, 9]

    is_ref_semposs = cl_args.ref_dataset_name == 'semanticPOSS'
    train_dl, train_dataset = get_data_loader(opt, 'train', opt.training.batch_size,is_ref_semposs=is_ref_semposs)
    val_dl, val_dataset = get_data_loader(opt, 'val' if (opt.training.isTrain or cl_args.on_input)  else 'test', opt.training.batch_size, shuffle=False, is_ref_semposs=is_ref_semposs)  
    test_dl, test_dataset = get_data_loader(opt, 'test', opt.training.batch_size, dataset_name=cl_args.ref_dataset_name, two_dataset_enabled=False, is_ref_semposs=is_ref_semposs)
    with torch.no_grad():
        seg_model = Segmentator(dataset_name=cl_args.ref_dataset_name if cl_args.seg_cfg_path == '' else 'synth', cfg_path=cl_args.seg_cfg_path).to(device)
    model = create_model(opt, lidar_A, None)      # create a model given opt.model and other options
    model.set_seg_model(seg_model)               # regular setup: load and print networks; create schedulers
    ## initilisation of the model for netF in cut
    train_dl_iter = iter(train_dl); 
    data = next(train_dl_iter)
    model.setup(opt.training)
    fid_cls = FID(seg_model, train_dataset, cl_args.ref_dataset_name, lidar_A) \
        if (cl_args.ref_dataset_name!= '' and 'reflectance' in opt.model.modality_B) else None
    fpd_cls = FPD(train_dataset, cl_args.ref_dataset_name, lidar_A) if cl_args.ref_dataset_name!= '' else None
    n_test_batch = 2 if cl_args.fast_test else  len(test_dl)
    test_dl_iter = iter(test_dl)
    data_dict = defaultdict(list)
    N = 2 * opt.training.batch_size if cl_args.fast_test else min(len(test_dataset), len(val_dataset), 1000)
    test_tq = tqdm.tqdm(total=N//opt.training.batch_size, desc='real_data', position=5)
    for i in range(0, N, opt.training.batch_size):
        data = next(test_dl_iter)
        data = fetch_reals(data, lidar_ref, device, opt.model.norm_label)
        data_dict['real-2d'].append(data['depth'])
        data_dict['real-3d'].append(depth_to_xyz(data['depth'], lidar_ref)[0])
        test_tq.update(1)
    epoch_tq = tqdm.tqdm(total=opt.training.n_epochs, desc='Epoch', position=1)
    start_from_epoch = model.schedulers[0].last_epoch if opt.training.continue_train else 0 
        
    #### Train & Validation Loop
    for epoch in range(start_from_epoch, opt.training.n_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        e_steps = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # Train loop
        if opt.training.isTrain:
            model.train(True)
            train_dl_iter = iter(train_dl)
            n_train_batch = 2 if cl_args.fast_test else len(train_dl)
            train_tq = tqdm.tqdm(total=n_train_batch, desc='Iter', position=3)
            for i in range(n_train_batch):  # inner loop within one epoch
                data = next(train_dl_iter)
                # import matplotlib.pyplot as plt
                # plt.figure(0)
                # plt.imshow(np.clip(data['depth'][0,0].numpy()* 5, 0, 1))
                # plt.figure(1)
                # plt.imshow(np.clip(data['reflectance'][0,0].numpy(),0 ,1))
                # plt.figure(2)
                # plt.imshow(data['label'][0,0].numpy())
                # plt.show()
                iter_start_time = time.time()  # timer for computation per iteration
                g_steps += 1
                e_steps += 1
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                if g_steps % opt.training.display_freq == 0:   # display images on visdom and save images to a HTML file
                    current_visuals = model.get_current_visuals()
                    visualizer.display_current_results('train',current_visuals, g_steps,ds_cfg, opt.dataset.dataset_A.name, lidar_A)

                if g_steps % opt.training.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    visualizer.print_current_losses('train', epoch, e_steps, losses, train_tq)
                    visualizer.plot_current_losses('train', epoch, losses, g_steps)
                    visualizer.writer.add_scalar('lr', model.schedulers[0].get_last_lr()[0], g_steps)
                if g_steps % opt.training.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    train_tq.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, g_steps))
                    model.save_networks('latest')
                train_tq.update(1)
        val_dl_iter = iter(val_dl)
        n_val_batch = 2 if cl_args.fast_test else  len(val_dl)

        ##### validation
        val_losses = defaultdict(list)
        model.train(False)
        tag = 'val' if opt.training.isTrain else 'test'
        val_tq = tqdm.tqdm(total=n_val_batch, desc='val_Iter', position=5)
        dis_batch_ind = np.random.randint(0, n_val_batch)
        data_dict['synth-2d'] = [] 
        data_dict['synth-3d'] = []
        fpd_points = []
        fid_samples = [] if fid_cls is not None else None
        iou_list = []
        m_acc_list = []
        rec_list = []
        prec_list = []
        label_map = ds_cfg_ref.kitti_to_POSS_map if is_ref_semposs and cl_args.map_label else None

        for i in range(n_val_batch):
            data = next(val_dl_iter)
            model.set_input(data)
            model.validate()
            model.calc_supervised_metrics(cl_args.no_inv, lidar_A, lidar, is_transformer)
            
            fetched_data = fetch_reals(data, lidar_A, device)
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
            data_dict['synth-2d'].append(synth_depth)
            ds_xyz , xyz = depth_to_xyz(synth_depth, lidar)
            data_dict['synth-3d'].append(ds_xyz)
            fpd_points.append(xyz)
            if fid_cls is not None and len(fid_samples) < cl_args.n_fid:
                synth_depth = tanh_to_sigmoid(synth_depth)
                synth_points = lidar.depth_to_xyz(tanh_to_sigmoid(synth_depth)) * lidar.max_depth
                synth_reflectance = tanh_to_sigmoid(synth_reflectance)
                synth_data = torch.cat([synth_depth, synth_points, synth_reflectance, synth_mask], dim=1)
                fid_samples.append(synth_data)
                if not opt.training.isTrain:
                    iou, m_acc, prec, rec = compute_seg_accuracy(seg_model, synth_data * fetched_data['mask'] , fetched_data['lwo'], ignore=ignore_label, label_map=label_map)
                    iou_list.append(iou.cpu().numpy())
                    m_acc_list.append(m_acc.cpu().numpy())
                    prec_list.append(prec.cpu().numpy())
                    rec_list.append(rec.cpu().numpy())

            if i == dis_batch_ind:
                current_visuals = model.get_current_visuals()
                visualizer.display_current_results(tag, current_visuals, g_steps, ds_cfg, opt.dataset.dataset_A.name, lidar_A)

            for k ,v in model.get_current_losses(is_eval=True).items():
                val_losses[k].append(v)
            val_tq.update(1)
        if not opt.training.isTrain:
            avg_m_acc = np.array(m_acc_list).mean()
            iou_avg = np.array(iou_list).mean(axis=0)
            prec_avg = np.array(prec_list).mean(axis=0)
            rec_avg = np.array(rec_list).mean(axis=0)
            label_names = seg_model.learning_class_to_label_name(np.arange(len(iou_avg)), ds_cfg_ref)
            print('avg seg acc:', np.round(avg_m_acc, 2))
            print('iou avg:')
            print_str = ''
            cross_class_iou_avg = iou_avg[iou_avg != 0.0].mean()
            for i, (l, iou) in enumerate(zip(label_names, iou_avg)):
                if iou > 0.0:
                    print_str = print_str + f'{l}:{np.round(iou, 4)} precision:{np.round(prec_avg[i],4)}, recall:{np.round(rec_avg[i],4)} '
            print(print_str)
            print('cross-class iou:', np.round(cross_class_iou_avg, 2))
        losses = {k: float(np.array(v).mean()) for k , v in val_losses.items()}
        visualizer.plot_current_losses(tag, epoch, losses, g_steps)
        visualizer.print_current_losses(tag, epoch, e_steps, losses, val_tq)
        
        ##### calculating unsupervised metrics


        for k ,v in data_dict.items():
            if isinstance(v, list):
                data_dict[k] = torch.cat(v, dim=0)[: N]
        scores = {}
        scores.update(compute_swd(data_dict["synth-2d"], data_dict["real-2d"]))
        scores["jsd"] = compute_jsd(data_dict["synth-3d"] / 2.0, data_dict["real-3d"] / 2.0)
        scores.update(compute_cov_mmd_1nna(data_dict["synth-3d"], data_dict["real-3d"], 512, ("cd",)))
        torch.cuda.empty_cache()
        if fid_cls is not None:
            scores['fid'] = fid_cls.fid_score(torch.cat(fid_samples, dim=0))
        if fpd_cls is not None:
            scores['fpd'] = fpd_cls.fpd_score(torch.cat(fpd_points, dim=0))
        if losses["depth/rmse"] < min_rmse and opt.training.isTrain:
            min_rmse = losses["depth/rmse"]
            model.save_networks('best')
        visualizer.plot_current_losses('unsupervised_metrics', epoch, scores, g_steps)
        visualizer.print_current_losses('unsupervised_metrics', epoch, e_steps, scores, val_tq)
        if opt.training.isTrain:
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        epoch_tq.update(1)
        print('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))


if __name__ == '__main__':
    main()
    
 