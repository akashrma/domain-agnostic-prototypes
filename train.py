import os
import json
import torch
import random

import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from loss import dice_loss, fixed_etf_loss

from sklearn.metrics import confusion_matrix
from skimage.exposure import match_histograms
from eval import eval_supervised, eval_validation

from utils import update_class_prototypes, log_losses_tensorboard, print_losses
from utils import compute_prf1, generate_pseudo_label, label_downsample, to_numpy
from utils import dice_eval, adjust_learning_rate, loss_calc, generate_etf_class_prototypes
from utils import compute_pairwise_cosines_std_and_shifted_mean, get_batch_class_centers


plt.switch_backend("agg")
interp_up = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

class PixelLevelFeatureProjector(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=256):
        super(PixelLevelFeatureProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C) 
        x = self.mlp(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        return x

def train_supervised(model, train_loader, val_loader, args):
    '''
    Supervised training for a single domain.
    '''
    input_size = args.input_size_source
    viz_tensorboard = os.path.exists(args.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    device = torch.device(args.device)
    
    model.train()
    model.to(device)
    # model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.optim_parameters(args.learning_rate),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.optim_parameters(args.learning_rate),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.optim_parameters(args.learning_rate),
                             lr=args.learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    
    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    
    train_loader_iter = enumerate(train_loader)
    val_loader_iter = enumerate(val_loader)

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    best_mean_dice = 0

    # lists for logging
    iterations = []
    std_cosine_feature_center_list = []
    avg_shifted_cos_feature_center_list = []
    
    log_file_path = Path(args.snapshot_dir) / "training_logs.json"
    log_data = []
    
    for i_iter in tqdm(range(args.max_iters + 1)):
        
        model.train()
        optimizer.zero_grad()

        adjust_learning_rate(optimizer, i_iter, writer, args)

        try:
            _, batch = train_loader_iter.__next__()
        except StopIteration:
            train_loader_iter = enumerate(train_loader)
            _, batch = train_loader_iter.__next__()
        images, labels, _ = batch

        # images = images.cuda()
        # labels = labels.cuda()
        images = images.to(args.device)
        labels = labels.to(args.device)

        features, pred_aux, pred_main = model(images)
        
        if args.multi_level_train:
            pred_aux = interp(pred_aux)
            loss_seg_aux = loss_calc(pred_aux, labels, args)
            loss_dice_aux = dice_loss(pred_aux, labels)
        else:
            loss_seg_aux = 0
            loss_dice_aux = 0
        
        pred_main = interp(pred_main)
        loss_seg_main = loss_calc(pred_main, labels, args)
        loss_dice_main = dice_loss(pred_main, labels)
        loss_seg_all = (args.lambda_seg_main * loss_seg_main
                        + args.lambda_seg_aux * loss_seg_aux
                        + args.lambda_dice_main * loss_dice_main
                        + args.lambda_dice_aux * loss_dice_aux
        )

        loss_total = loss_seg_all
        loss_total.backward()

        optimizer.step()

        torch.cuda.empty_cache()

        if i_iter % 10 == 0 and i_iter != 0:
            _, feature_dim, feature_H, feature_W = features.size()
            downsampled_labels = label_downsample(labels.unsqueeze(1), feature_H, feature_W) # B*feature_H*feature_W
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, feature_dim) # Shape: [B*H*W, feat_dim]
            downsampled_labels_flat = downsampled_labels.view(-1) # Shape: [B*H*W]
            class_centers = get_batch_class_centers(features_flat, downsampled_labels_flat, args.num_classes)
            std_cos_feat_center, avg_shift_cos_feat_center = compute_pairwise_cosines_std_and_shifted_mean(class_centers)

            # Save logs
            iterations.append(i_iter)
            std_cosine_feature_center_list.append(std_cos_feat_center)
            avg_shifted_cos_feature_center_list.append(avg_shift_cos_feat_center)


            current_losses = {'src_loss_seg_aux': loss_seg_aux,
                              'src_loss_seg_main': loss_seg_main,
                              'src_loss_dice_aux': loss_dice_aux,
                              'src_loss_dice_main': loss_dice_main,
                              'src_loss_seg_all': loss_seg_all
            }

            print(f"Iteration {i_iter}: {current_losses}")

            if viz_tensorboard:
                writer.add_scalar("StdCosine/FeatureCenters", std_cos_feat_center, i_iter)
                writer.add_scalar("ShiftedCosine/FeatureCenters", avg_shift_cos_feat_center, i_iter)
                log_losses_tensorboard(writer, current_losses, i_iter)

            log_entry = {"iteration": i_iter,
                        "std_cosine_feature_center": std_cos_feat_center,
                        "avg_shifted_cos_feature_center": avg_shift_cos_feat_center,
                        "src_loss_seg_aux": loss_seg_aux,
                        "src_loss_seg_main": loss_seg_main,
                        'src_loss_dice_aux': loss_dice_aux,
                        'src_loss_dice_main': loss_dice_main,
                        'src_loss_seg_all': loss_seg_all}

            log_data.append(log_entry)

            with open(log_file_path, "w") as json_file:
                json.dump(log_data, json_file, indent=4)

        # Validation
        # if i_iter % 10 == 0 and i_iter != 0:
            # print_losses(current_losses, i_iter)

            # classifier_weight = model.classifier.weight.data

            # dice_mean, _, _, _ = eval_validation(val_loader, model, "cuda", best_mean_dice, i_iter, args, writer)
            # best_mean_dice = max(np.mean(dice_mean), best_mean_dice)            
            # try:
            #     _, batch = val_loader_iter.__next__()
            # except StopIteration:
            #     val_loader_iter = enumerate(val_loader)
            #     _, batch = val_loader_iter.__next__()
            # images_val, labels_val, _ = batch

            # iter_eval_supervised(model, images_val, labels, labels_val, args)

        if i_iter % 100 == 0 and i_iter != 0:
            
            test_list_pth = args.testfile_path

            with open(test_list_pth) as fp:
                rows = fp.readlines()
            testfile_list = [row[:-1] for row in rows]

            dice_mean, dice_std, assd_mean, assd_std = eval_supervised(testfile_list, model, args.test_target)
            is_best = np.mean(dice_mean) > best_mean_dice
            best_mean_dice = max(np.mean(dice_mean),best_mean_dice)
            if is_best:
                print('Dice:')
                print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
                print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
                print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
                print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
                print('Mean:%.1f' % np.mean(dice_mean))
                print('ASSD:')
                print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
                print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
                print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
                print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
                print('Mean:%.1f' % np.mean(assd_mean))
                print('taking snapshot ...')
                print('exp =', args.snapshot_dir)
                snapshot_dir = Path(args.snapshot_dir)
                torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

            model.train()

            with open(log_file_path, "w") as json_file:
                json.dump(log_data, json_file, indent=4)
            

def train_supervised_etf(model, train_loader, val_loader, args):
    '''
    Supervised training for a single domain with neural collapse/ETF fixed prototypes.
    '''
    input_size = args.input_size_source
    viz_tensorboard = os.path.exists(args.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    device = torch.device(args.device)
    
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    projector = PixelLevelFeatureProjector(in_dim=2048, out_dim=256)
    projector = projector.to(device)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.optim_parameters(args.learning_rate),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.optim_parameters(args.learning_rate),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.optim_parameters(args.learning_rate),
                             lr=args.learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)

    optimizer_proj = optim.Adam(projector.parameters(),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)

    # for param in projector.parameters():
    #     param.requires_grad = False
    
    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    train_loader_iter = enumerate(train_loader)
    val_loader_iter = enumerate(val_loader)

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    class_prototypes = None
    best_mean_dice = 0

    # lists for logging
    iterations = []
    std_cosine_feature_center_list = []
    avg_shifted_cos_feature_center_list = []
    
    log_file_path = Path(args.snapshot_dir) / "training_logs.json"
    log_data = []
    
    for i_iter in tqdm(range(args.max_iters + 1)):
        
        model.train()
        projector.train()
        optimizer.zero_grad()
        optimizer_proj.zero_grad()

        adjust_learning_rate(optimizer, i_iter, writer, args)

        try:
            _, batch = train_loader_iter.__next__()
        except StopIteration:
            train_loader_iter = enumerate(train_loader)
            _, batch = train_loader_iter.__next__()
        images, labels, _ = batch

        images = images.to(args.device)
        labels = labels.to(args.device)
        
        features, pred_aux, pred_main = model(images.to())
        projected_features = projector(features.to(args.device))
        
        if args.multi_level_train:
            pred_aux = interp(pred_aux)
            loss_seg_aux = loss_calc(pred_aux, labels, args)
            loss_dice_aux = dice_loss(pred_aux, labels)
        else:
            loss_seg_aux = 0
            loss_dice_aux = 0
        
        pred_main = interp(pred_main)
        loss_seg_main = loss_calc(pred_main, labels, args)
        loss_dice_main = dice_loss(pred_main, labels)
        loss_seg_all = (args.lambda_seg_main * loss_seg_main
                        + args.lambda_seg_aux * loss_seg_aux
                        + args.lambda_dice_main * loss_dice_main
                        + args.lambda_dice_aux * loss_dice_aux
        )

        _, feature_dim, feature_H, feature_W = projected_features.size()
        
        downsampled_labels = labels_downsample(labels.unsqueeze(1), feature_H, feature_W) # B*feature_H*feature_W
        if class_prototypes is None:
            class_prototypes = generate_etf_class_prototypes(feature_dim, args.num_classes)
        
        loss_etf = fixed_etf_loss(projected_features, downsampled_labels, class_prototypes, None, args)
        loss_total = loss_seg_all + loss_etf
        loss_total.backward()

        optimizer.step()

        torch.cuda.empty_cache()

        if i_iter % 10 == 0 and i_iter != 0:
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, feature_dim) # Shape: [B*H*W, feat_dim]
            downsampled_labels_flat = downsampled_labels.view(-1) # Shape: [B*H*W]

            class_centers = get_batch_class_centers(feature_flat, downsampled_labels_flat, args.num_classes)
            std_cos_feat_center, avg_shift_cos_feat_center = compute_pairwise_cosines_std_and_shifted_mean(class_centers)

            # Save logs
            iterations.append(i_iter)
            std_cosine_feature_center_list.append(std_cos_feat_center)
            avg_shifted_cos_feature_center_list.append(avg_shift_cos_feat_center)


            current_losses = {'src_loss_seg_aux': loss_seg_aux,
                              'src_loss_seg_main': loss_seg_main,
                              'src_loss_dice_aux': loss_dice_aux,
                              'src_loss_dice_main': loss_dice_main,
                              'src_loss_seg_all': loss_seg_all,
                              'src_loss_etf': loss_etf,
            }

            print(f"Iteration {i_iter}: {current_losses}")

            if viz_tensorboard:
                writer.add_scalar("StdCosine/FeatureCenters", std_cos_feat_center, i_iter)
                writer.add_scalar("ShiftedCosine/FeatureCenters", avg_shift_cos_feat_center, i_iter)
                log_losses_tensorboard(writer, current_losses, i_iter)

            log_entry = {"iteration": i_iter,
                        "std_cosine_feature_center": std_cos_feat_center,
                        "avg_shifted_cos_feature_center": avg_shift_cos_feat_center,
                        "src_loss_seg_aux": loss_seg_aux,
                        "src_loss_seg_main": loss_seg_main,
                        'src_loss_dice_aux': loss_dice_aux,
                        'src_loss_dice_main': loss_dice_main,
                        'src_loss_seg_all': loss_seg_all,
                        'src_loss_etf': loss_etf}

            log_data.append(log_entry)

            with open(log_file_path, "w") as json_file:
                json.dump(log_data, json_file, indent=4)

        # Validation
        # if i_iter % 10 == 0 and i_iter != 0:
        #     print_losses(current_losses, i_iter)

            # try:
            #     _, batch = val_loader_iter.__next__()
            # except StopIteration:
            #     val_loader_iter = enumerate(val_loader)
            #     _, batch = val_loader_iter.__next__()
            # images_val, labels_val, _ = batch

            # iter_eval_supervised(model, images_val, labels, labels_val, args)

        if i_iter % 100 == 0 and i_iter != 0:
            
            test_list_pth = args.testfile_path

            with open(test_list_pth) as fp:
                rows = fp.readlines()
            testfile_list = [row[:-1] for row in rows]

            dice_mean, dice_std, assd_mean, assd_std = eval_supervised(testfile_list, model, args.test_target)
            is_best = np.mean(dice_mean) > best_mean_dice
            best_mean_dice = max(np.mean(dice_mean),best_mean_dice)
            if is_best:
                print('Dice:')
                print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
                print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
                print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
                print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
                print('Mean:%.1f' % np.mean(dice_mean))
                print('ASSD:')
                print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
                print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
                print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
                print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
                print('Mean:%.1f' % np.mean(assd_mean))
                print('taking snapshot ...')
                print('exp =', args.snapshot_dir)
                snapshot_dir = Path(args.snapshot_dir)
                torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

            model.train()

            with open(log_file_path, "w") as json_file:
                json.dump(log_data, json_file, indent=4)
            

def train_uda_dap(model, source_train_loader, target_train_loader, args):
    '''
    Unsupervised Domain Adaptation with Domain Agnostic Prototypes and Self-Labelling
    '''
    
    input_size = args.input_size_source
    viz_tensorboard = os.path.exists(args.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    source_class_prototypes = np.load(args.source_class_prototypes_init).squeeze()
    source_class_prototypes = torch.from_numpy(source_class_prototypes).float().cuda() # [num_classes, feat_dim]

    model.train()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.optim_parameters(args.learning_rate),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.optim_parameters(args.learning_rate),
                               lr=args.learning_rate,
                               betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.optim_parameters(args.learning_rate),
                             lr=args.learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    source_train_loader_iter = enumerate(source_train_loader)
    # source_val_loader_iter = enumerate(source_val_loader)
    target_train_loader_iter = enumerate(target_train_loader)
    # target_val_loader_iter = enumerate(target_val_loader)

    etf_class_prototypes = None
    best_mean_dice = 0

    for i_iter in tqdm(range(args.max_iters + 1)):

        model.train()
        optimizer.zero_grad()

        adjust_learning_rate(optimizer, i_iter, writer, args)

        try:
            _, batch = source_train_loader_iter.__next__()
        except StopIteration:
            source_train_loader_iter = enumerate(source_train_loader)
            _, batch = source_train_loader_iter.__next__()
        source_images, source_labels, _ = batch

        # SUPERVISED TRAINING FOR SOURCE IMAGES
        source_features, source_pred_aux, source_pred_main = model(source_images.cuda()) # [B, feat_dim, 33, 33], [B, num_classes, 33, 33], [B, num_classes, 33, 33]  

        if args.multi_level_train:
            source_pred_aux = interp(source_pred_aux) # [B, feat_dim, 256, 256]
            source_loss_seg_aux = loss_calc(source_pred_aux, source_labels, args)
            source_loss_dice_aux = dice_loss(source_pred_aux, source_labels)
        else:
            source_loss_seg_aux = 0
            source_loss_dice_aux = 0

        source_pred_main = interp(source_pred_main) # [B, feat_dim, 256, 256]
        source_loss_seg_main = loss_calc(source_pred_main, source_labels, args) # xent
        source_loss_dice_main = dice_loss(source_pred_main, source_labels)

        loss_seg_all = (args.lambda_seg_main * source_loss_seg_main
                        + args.lambda_seg_aux * source_loss_seg_aux
                        + args.lambda_dice_main * source_loss_dice_main
                        + args.lambda_dice_aux * source_loss_dice_aux)

        if i_iter > args.warmup_iter:

            try:
                _, batch = target_train_loader_iter.__next__()
            except StopIteration:
                target_train_loader_iter = enumerate(target_train_loader)
                _, batch = target_train_loader_iter.__next__()

            target_images, target_images_aug_list, _ = batch

            _, feature_dim, feature_H, feature_W = source_features.size()

            source_downsampled_labels = labels_downsample(source_labels.unsqueeze(1), feature_H, feature_W) # [B, 33, 33]
            if etf_class_prototypes is None:
                etf_class_prototypes = generate_etf_class_prototypes(feature_dim, args.num_classes) # [feat_dim, num_classes]

            print('Calculating source ETF loss...')
            print("*"*100)
            source_loss_etf = fixed_etf_loss(source_features, source_downsampled_labels, etf_class_prototypes, None, args) # why after normalization, we are getting higher loss? -> plot the softmax function with norm scaling

            source_class_prototypes = update_class_prototypes(source_features, source_downsampled_labels, source_class_prototypes, args) # [feat_dim, num_classes]
            
            target_feature_list = []
            target_pred_main_list = []
            target_features, _, target_pred_main = model(target_images.cuda())
            target_feature_list.append(target_features)
            # target_pred_main_list.append()

            for image_aug in target_images_aug_list:
                target_aug_features, _ , _ = model(image_aug.cuda())
                target_feature_list.append(target_aug_features)

            print('Calculating target ETF loss...')
            print("*"*100)
            target_loss_etf = 0
            for target_feature in target_feature_list:
                target_feature = interp(target_feature)
                hard_pixel_label, pixel_mask = generate_pseudo_label(target_feature, source_class_prototypes, args)
                print(hard_pixel_label.shape)
                print(pixel_mask.shape)
                target_loss_etf += fixed_etf_loss(target_feature, hard_pixel_label, etf_class_prototypes, pixel_mask, args)

            loss_etf_all = source_loss_etf + (args.target_etf_loss_weight * target_loss_etf)
        else:
            source_loss_etf = 0
            target_loss_etf = 0
            loss_etf_all = 0
        
        loss_total = loss_seg_all + loss_etf_all
        loss_total.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        current_losses = {'source_loss_seg_aux': source_loss_seg_aux,
                          'source_loss_seg_main': source_loss_seg_main,
                          'source_loss_dice_aux': source_loss_dice_aux,
                          'source_loss_dice_main': source_loss_dice_main,
                          'source_loss_etf': source_loss_etf,
                          'target_loss_etf': target_loss_etf,
                          'loss_etf_all': loss_etf_all,
        }

        # Validation
        if i_iter % 10 == 0 and i_iter != 0:
            print_losses(current_losses, i_iter)

            # try:
            #     _, batch = val_loader_iter.__next__()
            # except StopIteration:
            #     val_loader_iter = enumerate(val_loader)
            #     _, batch = val_loader_iter.__next__()
            # images_val, labels_val, _ = batch

            # iter_eval_supervised(model, images_val, labels, labels_val, args)

        if i_iter % 100 == 0 and i_iter != 0:
            
            test_list_pth = args.testfile_path

            with open(test_list_pth) as fp:
                rows = fp.readlines()
            testfile_list = [row[:-1] for row in rows]

            dice_mean, dice_std, assd_mean, assd_std = eval_supervised(testfile_list, model, args.test_target)
            is_best = np.mean(dice_mean) > best_mean_dice
            best_mean_dice = max(np.mean(dice_mean),best_mean_dice)
            if is_best:
                print('Dice:')
                print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
                print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
                print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
                print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
                print('Mean:%.1f' % np.mean(dice_mean))
                print('ASSD:')
                print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
                print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
                print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
                print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
                print('Mean:%.1f' % np.mean(assd_mean))
                print('taking snapshot ...')
                print('exp =', args.snapshot_dir)
                snapshot_dir = Path(args.snapshot_dir)
                torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

            model.train()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)