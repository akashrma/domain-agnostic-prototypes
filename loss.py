import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def cross_entropy_2d(pred, label, args):
    '''
    pred: B*C*H*W
    label: B*H*W
    '''

    assert not label.requires_grad
    assert pred.dim() == 4
    assert label.dim() == 3
    assert pred.size(0) == label.size(0), f'{pred.size(0)}vs{label.size(0)}'
    assert pred.size(2) == label.size(1), f'{pred.size(2)}vs{label.size(1)}'
    assert pred.size(3) == label.size(2), f'{pred.size(3)}vs{label.size(2)}'

    B, C, H, W = pred.size()
    label = label.view(-1)
    class_count = torch.bincount(label, minlength=C).float()
    try:
        assert class_count.size(0) == 5
        new_class_count = class_count
    except:
        new_class_count = torch.zeros(5).cuda().float()
        new_class_count[:class_count_size(0)] = class_count

    weight = (1 - (new_class_count + 1) / label.size(0))
    # pred = pred.transpose(1,2).transpose(2,3).contiguous() # B*C*H*W -> B*H*C*W -> B*H*W*C
    pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
    loss = F.cross_entropy(input=pred, target=label, weight=weight)
    return loss

def dice_loss(pred, target):
    '''
    pred: B*C*H*W
    target: N*H*W
    '''

    B, C, H, W = pred.size()
    pred = pred.cuda()
    target = target.cuda()
    target_onehot = torch.zeros([B,C,H,W]).cuda()
    target = torch.unsqueeze(target, dim=1) # B*1*H*W
    target_onehot.scatter_(1, target, 1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim()  == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "Target must only contain zeros and ones."

    eps = 1e-5
    probs = F.softmax(pred,dim=1)
    num   = probs * target_onehot  # B,C,H,W--p*g
    num   = torch.sum(num, dim=3)  # B,C,H
    num   = torch.sum(num, dim=2)  # B,C

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # B,C,H
    den1 = torch.sum(den1, dim=2)  # B,C

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # B,C,H
    den2 = torch.sum(den2, dim=2)  # B,C

    dice = 2.0 * (num / (den1 + den2+eps))  # B,C

    dice_total =  torch.sum(dice) / dice.size(0)  # divide by B -> Batch Size
    return 1 - 1.0 * dice_total/5.0

    