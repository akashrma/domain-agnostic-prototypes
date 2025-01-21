import numpy as np
import torch
import torch.nn as nn
from loss import cross_entropy_2d
import torch.nn.functional as F
import torch.sparse as sparse
from skimage.exposure import match_histograms

def lr_poly(base_lr, curr_iter, max_iter, power):
    '''
    Poly LR Scheduler
    '''
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, args, learning_rate):
    lr = lr_poly(learning_rate, i_iter, args.max_iters, args.lr_poly_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, args):
    '''
    adjust learning rate for main segnet
    '''
    _adjust_learning_rate(optimizer, i_iter, args, args.learning_rate)

def cross_entropy_2d()

def loss_calc(pred, label, args):
    '''
    Cross Entropy Loss for Semantic Segmentation
    pred: B*C*H*W
    label: B*H*W
    '''
    label = label.long().cuda()
    return cross_entropy_2d(pred, label, args)


    