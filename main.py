import os
import pprint
import random
import warnings
import yaml
import torch
import argparse
import numpy as np
import os.path as osp
from torch.utils import data

from train import train_supervised, train_supervised_etf, train_uda_dap
from data_reader import CTDataset, MRDataset, CTDataset_aug, MRDataset_aug
from model.deeplabv2 import get_deeplab_v2

parser = argparse.ArgumentParser(description="Domain Agnostic Prototype Segmentation.")
parser.add_argument("--training_mode", choices=['supervised', 'supervised_etf', 'uda_dap'], type=str, help="train supervised mode or unsupervised domain adaptation.")
parser.add_argument("--train_domain", choices=['MR', 'CT'], type=str)
parser.add_argument("--tensorboard_log_dir", default='logs', type=str, help="Tensorboard log directory.")
parser.add_argument("--viz_rate", default=10, type=int, help="visualization rate for tensorboard")
parser.add_argument("--random_seed", default=1, type=int, help="random seed init for the experiment")
parser.add_argument("--backbone_model", default="DeepLabv2", type=str, help="Backbone model for segmentation.")
parser.add_argument("--num_classes", default=4, type=int, help="Number of pixel classes.")
parser.add_argument("--multi_level_train", default=False, type=bool, help="For DeepLabv2, if you want to enable multi-level training.")
parser.add_argument("--model_dir", default=None, type=str, help="model weights location.")
parser.add_argument("--source_train_dir", default=None, type=str, help="list of source training samples as a .txt file")
parser.add_argument("--source_val_dir", default=None, type=str, help="list of source validation samples as a .txt file")
parser.add_argument("--source_train_gt_dir", default=None, type=str, help="list of source training ground truth as a  .txt file")
parser.add_argument("--source_val_gt_dir", default=None, type=str, help="list of source validation ground truth as a  .txt file")
parser.add_argument("--target_train_dir", default=None, type=str, help="list of target training samples as a .txt file")
parser.add_argument("--target_val_dir", default=None, type=str, help="list of target validation samples as a .txt file")
parser.add_argument("--target_val_gt_dir", default=None, type=str, help="list of target validation ground truth as a .txt file")
parser.add_argument("--num_workers", default=4, type=int, help="Number of worker for dataloader.")
parser.add_argument("--optimizer", default='SGD', type=str, help="Optimizer for training.")
parser.add_argument("--learning-rate", default=2.0e-3, type=float, help='Optimizer learning rate.')
parser.add_argument("--momentum", default=0.9, type=float, help="Optimizer momentum.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay in SGD.")
parser.add_argument("--input_size_source", nargs='+', type=int)
parser.add_argument("--max_iters", default=2800, type=int, help="Maximum number of iterations for training.")
parser.add_argument("--lr_poly_power", default=2, type=int, help="Exponential power to adjust poly learning rate.")
parser.add_argument("--test_target", default='MR', type=str, help="Choose between 'MR' or 'CT'.")
parser.add_argument("--snapshot_dir", default='snapshot', type=str, help="Directory to store model snapshot.")
parser.add_argument("--testfile_path", type=str)
parser.add_argument("--lambda_seg_main", default=1.0, type=float)
parser.add_argument("--lambda_seg_aux", default=0.1, type=float)
parser.add_argument("--lambda_dice_main", default=1.0, type=float)
parser.add_argument("--lambda_dice_aux", default=0.1, type=float)
parser.add_argument("--etf_loss_weight", default=0.4, type=float)
parser.add_argument("--pl_mode", choices=["naive_thresholding", "adv_thresholding", "thresh_feat_consistency", "pixel_self_labeling_OT"], type=str)
parser.add_argument("--pixel_sel_thresh", default=0.4, type=int)
parser.add_argument("--target_etf_loss_weight", default=0.4, type=float)
parser.add_argument("--source_class_prototypes_init", type=str)
parser.add_argument("--source_prototype_momentum", default=0.01, type=float)
parser.add_argument("--warmup-iter", default=50, type=int)
args = parser.parse_args()

def main():

    _init_fn = None
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    def _init_fn(worker_id):
        np.random.seed(args.random_seed+worker_id)

    if args.backbone_model == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=args.num_classes, multi_level=args.multi_level_train)
        saved_state_dict = torch.load(args.model_dir, weights_only=True, map_location=torch.device("cpu"))
        if 'DeepLab_resnet_pretrained' in args.model_dir:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

        print('Model loaded from:{}'.format(args.model_dir))

    else:
        raise NotImplementedError(f"Not yet supported {arg.backbone_model}")

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    model = torch.compile(model)

    if args.training_mode == 'supervised':
        
        train_data = args.source_train_dir
        train_gt_data = args.source_train_gt_dir
        val_data = args.source_val_dir
        val_gt_data = args.source_val_gt_dir
        
        if args.train_domain == 'MR':
            train_dataset = MRDataset(data_pth=train_data, gt_pth=train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            val_dataset = MRDataset(data_pth=val_data, gt_pth=val_gt_data,
                                   img_mean=img_mean, transform=transforms)
        elif args.train_domain == 'CT':
            train_dataset = CTDataset(data_pth=train_data, gt_pth=train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            val_dataset = CTDataset(data_pth=val_data, gt_pth=val_gt_data,
                                   img_mean=img_mean, transform=transforms)
        train_loader = data.DataLoader(train_dataset,
                                      batch_size=args.num_workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)
        val_loader = data.DataLoader(val_dataset,
                                    batch_size=args.num_workers,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)
        print('Dataloading finished.')
        print(f'Supervised training for {args.train_domain}.')
        train_supervised(model, train_loader, val_loader, args)
    elif args.training_mode == 'supervised_etf':
        
        train_data = args.source_train_dir
        train_gt_data = args.source_train_gt_dir
        val_data = args.source_val_dir
        val_gt_data = args.source_val_gt_dir
        
        if args.train_domain == 'MR':
            train_dataset = MRDataset(data_pth=train_data, gt_pth=train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            val_dataset = MRDataset(data_pth=val_data, gt_pth=val_gt_data,
                                   img_mean=img_mean, transform=transforms)
        elif args.train_domain == 'CT':
            train_dataset = CTDataset(data_pth=train_data, gt_pth=train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            val_dataset = CTDataset(data_pth=val_data, gt_pth=val_gt_data,
                                   img_mean=img_mean, transform=transforms)
        train_loader = data.DataLoader(train_dataset,
                                      batch_size=args.num_workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)
        val_loader = data.DataLoader(val_dataset,
                                    batch_size=args.num_workers,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)
        print('Dataloading finished.')
        print(f'Supervised ETF training for {args.train_domain}.')
        train_supervised_etf(model, train_loader, val_loader, args)
        
    elif args.training_mode == 'uda_dap':
        
        source_train_data = args.source_train_dir
        source_train_gt_data = args.source_train_gt_dir
        source_val_data = args.source_val_dir
        source_val_gt_data = args.source_val_gt_dir
        target_train_data = args.target_train_dir
        target_val_data = args.target_val_dir
        target_val_gt_data = args.target_val_gt_dir
        
        if args.train_domain == 'MR':
            source_train_dataset = MRDataset(data_pth=source_train_data, gt_pth=source_train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            # #source_val_dataset = MRDataset(data_pth=source_val_data, gt_pth=source_val_gt_data,
            #                        img_mean=img_mean, transform=transforms)
            target_train_dataset = CTDataset_aug(data_pth=target_train_data,
                                                img_mean=img_mean, transform=transforms,
                                                aug_transform=True)
            # #target_val_dataset = CTDataset(data_pth=target_val_data, gt_pth=target_val_gt_data,
            #                               img_mean=img_mean, transform=transforms)
        elif args.train_domain == 'CT':
            source_train_dataset = CTDataset(data_pth=source_train_data, gt_pth=source_train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            # #source_val_dataset = CTDataset(data_pth=source_val_data, gt_pth=source_val_gt_data,
            #                        img_mean=img_mean, transform=transforms)
            target_train_dataset = MRDataset_aug(data_pth=target_train_data,
                                                img_mean=img_mean, transform=transforms,
                                                aug_transform=True)
            # #target_val_dataset = MRDataset(data_pth=target_val_data, gt_pth=target_val_gt_data,
            #                               img_mean=img_mean, transform=transforms)
        source_train_loader = data.DataLoader(source_train_dataset,
                                      batch_size=args.num_workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)
        # #source_val_loader = data.DataLoader(source_val_dataset,
        #                               batch_size=args.num_workers,
        #                               shuffle=True,
        #                               pin_memory=True,
        #                               worker_init_fn=_init_fn)
        target_train_loader = data.DataLoader(target_train_dataset,
                                      batch_size=args.num_workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)
        # target_val_loader = data.DataLoader(target_val_dataset,
        #                               batch_size=args.num_workers,
        #                               shuffle=True,
        #                               pin_memory=True,
        #                               worker_init_fn=_init_fn)
        print('Dataloading finished.')
        print(f'Domain Agnostic Prototype Training for {args.train_domain}.')
        train_uda_dap(model, source_train_loader, target_train_loader, args)

if __name__ == '__main__':
    main()
