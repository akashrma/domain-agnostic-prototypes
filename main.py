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

from train import train_supervised
from data_reader import CTDataset, MRDataset, CTDataset_aug, MRDataset_aug
from model.deeplabv2 import get_deeplab_v2

parser = argparse.ArgumentParser(description="Domain Agnostic Prototype Segmentation.")
parser.add_argument("--run_mode", choices=['train', 'test'], type=str, help="run main script for training or testing")
parser.add_argument("--training_mode", choices=['supervised', 'supervised_etf', 'uda_dap'], type=str, help="train supervised mode or unsupervised domain adaptation.")
parser.add_argument("--train_domain", choices=['MR', 'CT'], type=str)
parser.add_argument("--tensorboard_log_dir", default='logs', type=str, help="Tensorboard log directory.")
parser.add_argument("--viz_rate", default=10, type=int, help="visualization rate for tensorboard")
parser.add_argument("--random_seed", default=42, type=int, help="random seed init for the experiment")
parser.add_argument("--backbone_model", default="DeepLabv2", type=str, help="Backbone model for segmentation.")
parser.add_argument("--num_classes", default=4, type=int, help="Number of pixel classes.")
parser.add_argument("--multi_level_train", default=False, type=bool, help="For DeepLabv2, if you want to enable multi-level training.")
parser.add_argument("--model_dir", default=None, type=str, help="model weights location.")
parser.add_argument("--train_dir", default=None, type=str, help="list of training samples as a .txt file")
parser.add_argument("--val_dir", default=None, type=str, help="list of validation samples as a .txt file")
parser.add_argument("--train_gt_dir", default=None, type=str, help="list of training ground truth as a  .txt file")
parser.add_argument("--val_gt_dir", default=None, type=str, help="list of validation ground truth as a  .txt file")
parser.add_argument("--num_workers", default=4, type=int, help="Number of worker for dataloader.")
parser.add_argument("--optimizer", default='SGD', type=str, help="Optimizer for training.")
parser.add_argument("--learning-rate", default=1.0e-3, type=float, help='Optimizer learning rate.')
parser.add_argument("--momentum", default=0.9, type=float, help="Optimizer momentum.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay in SGD.")
parser.add_argument("--input_size_source", nargs='+', type=int)
parser.add_argument("--max_iters", default=1600, type=int, help="Maximum number of iterations for training.")
parser.add_argument("--lr_poly_power", default=2, type=int, help="Exponential power to adjust poly learning rate.")
parser.add_argument("--test_target", default='MR', type=str, help="Choose between 'MR' or 'CT'.")
parser.add_argument("--snapshot_dir", default='snapshot', type=str, help="Directory to store model snapshot.")
parser.add_argument("--testfile_path", type=str)
parser.add_argument("--lambda_seg_main", default=1.0, type=float)
parser.add_argument("--lambda_seg_aux", default=0.1, type=float)
parser.add_argument("--lambda_dice_main", default=1.0, type=float)
parser.add_argument("--lambda_dice_aux", default=0.1, type=float)
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

    train_data = args.train_dir
    train_gt_data = args.train_gt_dir
    val_data = args.val_dir
    val_gt_data = args.train_gt_dir

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    if args.training_mode == 'supervised':
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
        train_supervised(model, train_loader, val_loader, args)
    elif args.training_mode == 'supervised_etf':
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
        train_supervised_etf(model, train_loader, val_loader)
    elif args.training_mode == 'uda_dap':
        # NOT IMPLEMENTED YET
        if args.train_domain == 'MR':
            source_train_dataset = MRDataset(data_pth=train_data, gt_pth=train_gt_data,
                                     img_mean=img_mean, transform=transforms)
            source_val_dataset = MRDataset(data_pth=val_data, gt_pth=val_gt_data,
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
        train_dap_uda(model, source_train_loader, source_val_loader, target_train_loader)

if __name__ == '__main__':
    main()
