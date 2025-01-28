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
    return base_lr * ((1 - float(curr_iter) / max_iter) ** power)

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

def loss_calc(pred, label, args):
    '''
    Cross Entropy Loss for Semantic Segmentation
    pred: B*C*H*W
    label: B*H*W
    '''
    label = label.long().cuda()
    return cross_entropy_2d(pred, label, args)

def dice_eval(pred, label, n_class):
    '''
    pred: B*C*H*W
    label: B*H*W
    '''
    pred = torch.argmax(pred, dim=1) # B*H*W
    dice = 0
    dice_arr = []
    each_class_number = []

    eps = 1e-7

    for i in range(n_class):
        A = (pred == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        inse = torch.sum(A*B).float()
        union = (torch.sum(A) + torch.sum(B)).float()
        dice += 2*inse/(union + eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice, dice_arr, np.hstack(each_class_number)

def generate_random_orthogonal_matrix(feature_dim, num_classes):
    a = np.random.random(size=(feature_dim, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
    return P

def generate_etf_class_prototypes(feature_dim, num_classes):
    print(f"Generating ETF class prototypes for K={num_classes} and d={feature_dim}.")
    d = feature_dim
    K = num_classes
    P = generate_random_orthogonal_matrix(feature_dim=d, num_classes=K)
    I = torch.eye(K)
    one = torch.ones(K, K)
    M_star = np.sqrt(K / (K-1)) * torch.matmul(P, I-((1/K) * one))
    M_star = M_star.cuda()
    return M_star