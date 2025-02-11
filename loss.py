import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import logging

logging.basicConfig(level=logging.INFO, format="$(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

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

def fixed_etf_loss(features, labels, class_prototypes, reliable_pixel_mask, args):
    '''
    features: [B, feature_dim, H, W] -> B, 2048, 33, 33
    labels: [B, H, W] -> B, 33, 33
    class_prototypes: [feat_dim, num_classes]
    '''

    B, feature_dim, H, W = features.shape
    assert feature_dim == 256, "Model output's feature dimension should be 256."

    num_classes = args.num_classes
    # print(f'Norm value of the first pixel feature before normalization: {torch.norm(features[0,:,0,0], p=2)}.')
    # features = F.normalize(features, p=2, dim=1)
    # print(f'Norm value of the first pixel feature after normalization: {torch.norm(features[0,:,0,0], p=2)}.')
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, feature_dim) # Shape: [B*H*W, num_classes]
    labels_flat = labels.view(-1) # Shape: [B*H*W]

    original_pixel_num = len(labels_flat)
    
    if reliable_pixel_mask == None:
        reliable_pixel_mask = torch.ones(B*H*W)
        reliable_pixel_mask = reliable_pixel_mask.bool()

    features_flat = features_flat[reliable_pixel_mask] # [p1_2048, p2_2048,...,p4356_2048]
    labels_flat = labels_flat[reliable_pixel_mask] # [l1, l2,...., l4356] li belongs to {0,1,2,3,4}
 
    reduced_pixel_num = len(labels_flat)
    
    print(f'Number of reliable pixels have been reduced by {original_pixel_num - reduced_pixel_num}.')
    
    class_feature_sums = torch.zeros(num_classes, feature_dim, device=features.device) # Shape: [num_classes, 2048] 
    class_pixel_counts = torch.zeros(num_classes, device=features.device) # Shape: [num_classes] [0, 0, 0, 0, 0]

    # Compute feature sums and pixel counts
    for cls in range(num_classes):
        mask = (labels_flat == cls)
        if mask.sum() == 0:
            logger.info(f"No pixels founds for class {cls}, skipping.")
            continue

        class_feature_sums[cls] = features_flat[mask].sum(dim=0)
        class_pixel_counts[cls] = mask.sum()
        logger.info(f"Class {cls}: Pixel count: {class_pixel_counts[cls].item()}")

    class_feature_centers = class_feature_sums / (class_pixel_counts.view(-1, 1) + 1e-6)
    class_feature_centers = F.normalize(class_feature_centers, p=2, dim=-1)
    logger.info(f"Class feature centers: {class_feature_centers}")

    loss_feat_center = 0.0
    class_prototypes = class_prototypes.view(num_classes, feature_dim)

    for cls in range(num_classes):
        if class_pixel_counts[cls] == 0:
            continue

        sel_class_feature_center = class_feature_centers[cls]
        sel_class_prototype = class_prototypes[cls]

        dot_prods = torch.einsum('d,kd->k', sel_class_feature_center, class_prototypes)
        # max_dot_prod = dot_prods.max()
        # exp_dot_prods = torch.exp(dot_prods - max_dot_pro) 
        exp_dot_prods = torch.exp(dot_prods) 
        # numr = torch.exp(torch.dot(sel_class_feature_center, sel_class_prototype) - max_dot_pro)
        numr = torch.exp(torch.dot(sel_class_feature_center, sel_class_prototype))
        denr = exp_dot_prods.sum()

        softmax = numr / denr
        loss_feat_center += -torch.log(softmax.clamp(min=1e-12)) # Clamp to avoid log(0)
        logger.info(f"Class {cls}: Fixed ETF Loss contribution: {-torch.log(softmax.clamp(min=1e-12)).item()}")
    if loss_feat_center != 0.0:
        logger.info(f"Total Fixed ETF Loss: {loss_feat_center.item()}")
    else:
        logger.info(f"Total Fixed ETF Loss: {loss_feat_center}")

    return (args.etf_loss_weight * loss_feat_center)