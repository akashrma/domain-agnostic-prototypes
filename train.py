import os
from pathlib import Path
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from skimage.exposure import match_histograms
import torch.optim as optim
from tensorboardX import SummaryWriter

from torch import nn
from tqdm import tqdm
from torchvision.utils import make_grid
from utils import dice_eval, adjust_learning_rate, loss_calc, generate_etf_class_prototypes
from loss import dice_loss, fixed_etf_loss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from eval import eval_supervised

plt.switch_backend("agg")
interp_up = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

def computer_prf1(true_mask, pred_mask):
    """
    Compute precision, recall, and F1 metrics for predicted mask against ground truth
    """
    conf_mat = confusion_matrix(true_mask.reshape(-1), pred_mask.reshape(-1), labels=[False, True])
    p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
    r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return conf_mat, p, r, f1

def generate_psuedo_label(target_features, domain_agnostic_prototypes, mode, args):
    '''
    B: batch size
    feat_dim: feature dimension
    H: Height
    W: Width
    mode: Choose out of three options - ["thresholding", "thresh_feat_consistency", "pixel_self_labeling_OT"]
    
    target_features: B*feat_dim*H*W
    domain_agnostic_prototypes: C*feat_dim

    domain_agnostic_prototypes are already normalized.
    '''
    if mode == 'thresholding':
        target_features_detach = target_features.detach()
        batch, feat_dim, H, W = target_features_detach.size()
        target_features_detach = interp_up(target_features_detach)
        target_features_detach = F.normalize(target_features_detach, p=2, dim=1)
        target_features_detach = target_features_detach.permute(0, 2, 3, 1) # B*H*W*feat_dim
        domain_agnostic_prototypes = domain_agnostic_prototypes.transpose(0, 1) # feat_dim*C

        batch_pixel_cosine_sim = torch.matmul(target_feature_detach, domain_agnostic_prototypes)
        threshold = args.pixel_sel_thresh
        pixel_mask = pixel_selection(batch_pixel_cosine_sim, threshold)
        hard_pixel_mask = torch.argmax(batch_pixel_cosine_dim, dim=1)

        return hard_pixel_mask, pixel_mask
        
    elif mode == 'thresh_feat_consistency':
        raise NotImplementedError(f"Not yet supported {mode} for generating pseudo labels.")
    elif mode == 'pixel_self_labeling_OT':
        raise NotImplementedError(f"Not yet supported {mode} for generating pseudo labels.")
    else:
        raise ValueError(f"Input valid pseudo label generation methods. Valid choices: ['thresholding', 'thresh_feat_consistency', 'pixel_self_labeling_OT']")

def pixel_selection(batch_pixel_cosine_sim, threshold):

    batch_sort_cosine, _ = torch.sort(batch_pixel_cosine_sim, dim=1)
    pixel_sub_cosine = batch_sort_cosine[:, -1] - batch_sort_cosine[:, -2]
    pixel_mask = pixel_sub_cosine > threshold

    return pixel_mask

def iter_eval_supervised(model, images_val, labels, labels_val, args):
    model.eval()

    with torch.no_grad():
        num_classes = args.num_classes
        interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        features, pred_aux, pred_main = model(images_val.cuda())

        pred_main = interp(pred_main)
        _, val_dice_arr, val_class_number = dice_eval(pred=pred_main, label=labels_val.cuda(), n_class=num_classes)
        val_dice_arr = np.hstack(val_dice_arr)

        print('Dice Score')
        print('####### Validation Set #######')
        print('Each class number {}'.format(val_class_number))
        print('Myo:{:.3f}'.format(val_dice_arr[1]))
        print('LAC:{:.3f}'.format(val_dice_arr[2]))
        print('LVC:{:.3f}'.format(val_dice_arr[3]))
        print('AA:{:.3f}'.format(val_dice_arr[4]))
        print('####### Validation Set #######')
        
def label_downsample(labels, feat_h, feat_w):
    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=[feat_h, feat_w], mode='nearest')
    labels = labels.int()
    return labels

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name}={to_numpy(loss_value):.3f}')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter={i_iter} {full_string}')

def labels_downsample(labels, feature_H, feature_W):
    '''
    labels: B*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=[feature_H, feature_W], mode='nearest')
    labels = labels.int()
    return labels

def train_supervised(model, train_loader, val_loader, args):
    '''
    Supervised training for a single domain.
    '''
    input_size = args.input_size_source
    viz_tensorboard = os.path.exists(args.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    
    model.train()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.SGD(model.optim_parameters(args.learning_rate),
                             lr=args.learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    

    train_loader_iter = enumerate(train_loader)
    val_loader_iter = enumerate(val_loader)

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    best_mean_dice = 0
    for i_iter in tqdm(range(args.max_iters + 1)):
        
        model.train()
        optimizer.zero_grad()

        adjust_learning_rate(optimizer, i_iter, args)

        try:
            _, batch = train_loader_iter.__next__()
        except StopIteration:
            train_loader_iter = enumerate(train_loader)
            _, batch = train_loader_iter.__next__()
        images, labels, _ = batch

        features, pred_aux, pred_main = model(images.cuda())
        
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

        current_losses = {'loss_seg_aux': loss_seg_aux,
                          'loss_seg_main': loss_seg_main,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main': loss_dice_main, 
        }

        # Validation
        if i_iter % 10 == 0 and i_iter != 0:
            print_losses(current_losses, i_iter)

            try:
                _, batch = val_loader_iter.__next__()
            except StopIteration:
                val_loader_iter = enumerate(val_loader)
                _, batch = val_loader_iter.__next__()
            images_val, labels_val, _ = batch

            iter_eval_supervised(model, images_val, labels, labels_val, args)

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

def train_supervised_etf(model, train_loader, val_loader, args):
    '''
    Supervised training for a single domain with neural collapse/ETF fixed prototypes.
    '''
    input_size = args.input_size_source
    viz_tensorboard = os.path.exists(args.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    
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
    
    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    train_loader_iter = enumerate(train_loader)
    val_loader_iter = enumerate(val_loader)

    transforms = None
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    class_prototypes = None
    best_mean_dice = 0
    
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

        features, pred_aux, pred_main = model(images.cuda())
        
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

        _, feature_dim, feature_H, feature_W = features.size()
        
        downsampled_labels = labels_downsample(labels.unsqueeze(1), feature_H, feature_W) # B*feature_H*feature_W
        if class_prototypes is None:
            class_prototypes = generate_etf_class_prototypes(feature_dim, args.num_classes)
        
        loss_etf = fixed_etf_loss(features, downsampled_labels, class_prototypes, args)
        loss_total = loss_seg_all + loss_etf
        loss_total.backward()

        optimizer.step()

        torch.cuda.empty_cache()

        current_losses = {'loss_seg_aux': loss_seg_aux,
                          'loss_seg_main': loss_seg_main,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main': loss_dice_main,
                          'loss_etf': loss_etf,
        }

        # Validation
        if i_iter % 10 == 0 and i_iter != 0:
            print_losses(current_losses, i_iter)

            try:
                _, batch = val_loader_iter.__next__()
            except StopIteration:
                val_loader_iter = enumerate(val_loader)
                _, batch = val_loader_iter.__next__()
            images_val, labels_val, _ = batch

            iter_eval_supervised(model, images_val, labels, labels_val, args)

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

def train_uda_dap()