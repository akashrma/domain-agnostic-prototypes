import numpy as np
import torch
import torch.nn as nn
from loss import cross_entropy_2d
import torch.nn.functional as F
import torch.sparse as sparse
from skimage.exposure import match_histograms

def log_gradient_norms(model, writer, i_iter):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad_norm(2)
            total_norm += param_norm.item() ** 2
            writer.add_scalar(f'gradients/{name}', param_norm.item(), i_iter)

    total_norm = total_norm ** 0.5
    writer.add_scalar('gradients/total_norm', total_norm, i_iter)

def lr_poly(base_lr, curr_iter, max_iter, power):
    '''
    Poly LR Scheduler
    '''
    return base_lr * ((1 - float(curr_iter) / max_iter) ** power)

def adjust_learning_rate(optimizer, i_iter, writer, args):
    '''
    adjust learning rate for main segnet
    '''
    lr = lr_poly(args.learning_rate, i_iter, args.max_iters, args.lr_poly_power)
    optimizer.param_groups[0]['lr'] = lr
    writer.add_scalar('learning_rate_main', optimizer.param_groups[0]['lr'], i_iter)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        writer.add_scalar('learning_rate_classifier', optimizer.param_groups[1]['lr'], i_iter)

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

def computer_prf1(true_mask, pred_mask):
    """
    Compute precision, recall, and F1 metrics for predicted mask against ground truth
    """
    conf_mat = confusion_matrix(true_mask.reshape(-1), pred_mask.reshape(-1), labels=[False, True])
    p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
    r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return conf_mat, p, r, f1

def generate_pseudo_label(target_features, source_class_prototypes,  args):
    '''
    B: batch size
    feat_dim: feature dimension
    H: Height
    W: Width
    mode: Choose out of four options - ["thresholding", "thresh_feat_consistency", "pixel_self_labeling_OT"]
    
    target_features: [B, feat_dim, H, W]
    source_class_prototypes: [num_classes, feat_dim]

    '''
    if args.pl_mode == 'naive_thresholding':
        target_features_detach = target_features.detach() # [B, feat_dim, H, W] -> [B, 2048, 33, 33]
        batch, feat_dim, H, W = target_features_detach.size()
        # target_features_detach = interp_up(target_features_detach) # [B, feat_dim, H, W] -> [B, 2048, 256, 256]
        target_features_detach = F.normalize(target_features_detach, p=2, dim=1) 
        source_class_prototypes_normalized = F.normalize(source_class_prototypes, p=2, dim=1) # [num_classes, feat_dim]
        target_features_detach = target_features_detach.permute(0, 2, 3, 1) # [B, H, W, feat_dim]
        target_features_detach = torch.reshape(target_features_detach, [-1, feat_dim]) # [B*H*W, feat_dim]
        source_class_prototypes_normalized = source_class_prototypes_normalized.transpose(0, 1) # [feat_dim , num_classes]

        batch_pixel_cosine_sim = torch.matmul(target_features_detach, source_class_prototypes_normalized) # [B*H*W, num_classes]
        threshold = args.pixel_sel_thresh
        max_cosine_sim, _ = torch.max(batch_pixel_cosine_sim, dim=-1) 
        pixel_mask = max_cosine_sim > threshold # [B*H*W]
        hard_pixel_label = torch.argmax(batch_pixel_cosine_sim, dim=-1) # [B*H*W]

        return hard_pixel_label, pixel_mask
        
    elif args.pl_mode == 'adv_thresholding':
        target_features_detach = target_features.detach()
        batch, feat_dim, H, W = target_features_detach.size()
        target_features_detach = interp_up(target_features_detach)
        target_features_detach = F.normalize(target_features_detach, p=2, dim=1)
        source_class_prototypes_normalized = F.normalize(source_class_prototypes, p=2, dim=1)
        target_features_detach = target_features_detach.permute(0, 2, 3, 1) # [B, H, W, feat_dim]
        target_features_detach = torch.reshape(target_features_detach, [-1, feat_dim])
        source_class_prototypes_normalized = source_class_prototypes_normalized.transpose(0, 1) # [feat_dim , num_classes]

        batch_pixel_cosine_sim = torch.matmul(target_features_detach, source_class_prototypes_normalized)
        threshold = args.pixel_sel_thresh
        pixel_mask = pixel_selection(batch_pixel_cosine_sim, threshold)
        hard_pixel_label = torch.argmax(batch_pixel_cosine_sim, dim=-1)
        
        return hard_pixel_label, pixel_mask
        
    elif args.pl_mode == 'thresh_feat_consistency':
        raise NotImplementedError(f"Not yet supported {mode} for generating pseudo labels.")
    elif args.pl_mode == 'pixel_self_labeling_OT':
        raise NotImplementedError(f"Not yet supported {mode} for generating pseudo labels.")
    else:
        raise ValueError(f"Input valid pseudo label generation methods. Valid choices: ['thresholding', 'thresh_feat_consistency', 'pixel_self_labeling_OT']")

def pixel_selection(batch_pixel_cosine_sim, threshold):

    batch_sort_cosine, _ = torch.sort(batch_pixel_cosine_sim, dim=-1)
    pixel_sub_cosine = batch_sort_cosine[:,:,:,-1] - batch_sort_cosine[:,:,:,-2]
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

def update_class_prototypes(source_features, source_downsampled_labels, curr_source_class_prototypes, args):
    '''
    source_class_features: B*C*H*W
    source_labels        : B*H*W
    m                    : source class prototypes update momentum
    '''

    batch_features = source_features
    batch_label_downsampled = source_downsampled_labels.cuda()
    m = args.source_prototype_momentum

    batch_class_center_feat_list = []
    for i in range(args.num_classes):
        feature_mask = torch.eq(batch_label_downsampled, i).float().cuda()
        class_features = batch_features * feature_mask
        class_features_sum = torch.sum(class_features, [0, 2, 3])
        class_num = torch.sum(feature_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_feature = curr_source_class_prototypes[i, :].detach()
        else:
            batch_class_center_feature = class_features_sum / class_num
        batch_class_center_feature = batch_class_center_feature.unsqueeze(0)
        batch_class_center_feat_list.append(batch_class_center_feature)

    batch_class_center_features = torch.cat(batch_class_center_feat_list, dim=0)
    class_center_features_update = m * curr_source_class_prototypes.detach() + (1-m) * batch_class_center_features

    return class_center_features_update

#################
# Visualization #
#################

def save_as_gif(img_path, gif_save_path):
    images = []
    for img in glob.glob(img_path):
        images.append(iio.imread(img))
    iio.mimsave(gif_save_path, images)

def decet(feature, targets, i_iter, save_path):
    color = ["red", "black", "yellow", "green", "pink",
             "gray", "lightgreen", "orange", "blue", "teal"]
    cls_ids = [0, 1, 2, 3, 4]  # Adjust if more classes
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.ion()
    plt.clf()
    plt.figure(figsize=(6,6))
    
    feature_np = feature.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()
    
    for j in cls_ids:
        mask = (targets_np == j)
        feature_ = feature_np[mask]
        x = feature_[:, 0]
        y = feature_[:, 1]
        plt.plot(x, y, ".", color=color[j], label=f"Class {j}")
    plt.legend(loc="upper right")
    plt.title(f"Scatter Plot, epoch={i_iter+1}")
    plt.savefig(f'{save_path}/{i_iter+1}.jpg')
    plt.close()

def feature_dist(features, save_path, plot_title):
    frac = 0.3
    X, Y = np.mgrid[-1-frac:1+frac:int(100*(1+frac))*1j,
                    -1-frac:1+frac:int(100*(1+frac))*1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = stats.gaussian_kde(np.transpose(features))
    Z = np.reshape(kernel(positions).T, X.shape)

    fig, ax = plt.subplots(figsize=(6,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    sc = ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                   extent=[-1-frac, 1+frac, -1-frac, 1+frac])
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlim([-1-frac, 1+frac])
    ax.set_ylim([-1-frac, 1+frac])
    plt.title(plot_title)
    plt.savefig(save_path)
    plt.close()

def class_feature_dist(features, targets, i_iter, save_path):
    cls_ids = [0, 1, 2, 3, 4]  # Adjust if more classes
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    features = features.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()

    # Overall distribution
    all_feat_dist_path = os.path.join(save_path, 'all')
    if not os.path.exists(all_feat_dist_path):
        os.makedirs(all_feat_dist_path)
    feature_dist(features=features,
                 save_path=os.path.join(all_feat_dist_path, f'{i_iter+1}.jpg'),
                 plot_title=f'Feature Distribution, ep: {i_iter+1}')

    # Per-class distribution
    for j in cls_ids:
        mask = (targets_np == j)
        feature_ = features[mask]
        class_feat_dist_path = os.path.join(save_path, f'class_{j}')
        if not os.path.exists(class_feat_dist_path):
            os.makedirs(class_feat_dist_path)
        feature_dist(features=feature_,
                     save_path=os.path.join(class_feat_dist_path, f'{i_iter+1}.jpg'),
                     plot_title=f'Class {j}, ep: {i_iter+1}')

def vMF_angle(features, save_path, plot_title, y_lim):
    # features shape: [N, 2]
    # angles range: -pi to pi
    features_np = features.cpu().detach().numpy()
    angles = [np.arctan2(f[1], f[0]) for f in features_np]

    fig, ax = plt.subplots(figsize=(6,3))
    plt.xlim(-np.pi, np.pi)
    plt.ylim(0, y_lim)
    plt.title(plot_title)
    sns.histplot(angles, kde=True, bins=150, edgecolor="white", ax=ax)
    plt.savefig(save_path)
    plt.close()

def class_vMF_angle(features, targets, i_iter, save_path):
    cls_ids = [0, 1, 2, 3, 4]  # Adjust if more classes
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    features = features.cpu().detach()

    # Overall distribution
    all_angle_dist_path = os.path.join(save_path, 'all')
    if not os.path.exists(all_angle_dist_path):
        os.makedirs(all_angle_dist_path)
    vMF_angle(features=features,
              save_path=os.path.join(all_angle_dist_path, f'{i_iter+1}.jpg'),
              plot_title=f'Angle Distribution, ep: {i_iter+1}',
              y_lim=1000)

    # Per-class distribution
    targets_np = targets.cpu().detach().numpy()
    for j in cls_ids:
        mask = (targets_np == j)
        feature_ = features[mask]
        class_angle_dist_path = os.path.join(save_path, f'class_{j}')
        if not os.path.exists(class_angle_dist_path):
            os.makedirs(class_angle_dist_path)
        vMF_angle(features=feature_,
                  save_path=os.path.join(class_angle_dist_path, f'{i_iter+1}.jpg'),
                  plot_title=f'Class {j}, ep: {i_iter+1}',
                  y_lim=200)

def compute_pairwise_cosines_std_and_shifted_mean(weight_matrix):
    """
    Returns:
      std_cosine: standard deviation of pairwise cosines
      avg_shifted_cos: average of |cos + 1/(num_classes-1)| over all distinct pairs
    """
    # weight_matrix shape: [num_classes, feat_dim]
    # Normalize each row
    normalized = nn.functional.normalize(weight_matrix, p=2, dim=1)  # shape [num_classes, feat_dim]
    # Compute all pairwise cosines
    cos_matrix = torch.matmul(normalized, normalized.t())  # [num_classes, num_classes]
    # We only need the upper (or lower) triangular off-diagonal
    num_classes = cos_matrix.shape[0]
    triu_indices = torch.triu_indices(num_classes, num_classes, offset=1)
    pairwise_cosines = cos_matrix[triu_indices[0], triu_indices[1]]

    std_cosine = torch.std(pairwise_cosines)

    # Shifted cos: | cos(i,j) + 1/(C-1) |
    shifted_values = torch.abs(pairwise_cosines + 1.0/(num_classes-1))
    avg_shifted_cos = torch.mean(shifted_values)

    return std_cosine.item(), avg_shifted_cos.item()

def get_batch_class_centers(features, labels, num_classes):
    """
    Compute mean feature center per class from the batch.
    N := Number of valid pixels in the batch.
    features: [N, feat_dim]
    labels:   [N]
    Returns: [num_classes, feat_dim]
    """
    device = features.device
    feat_dim = features.shape[1]

    centers = torch.zeros(num_classes, feat_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            centers[c] = features[mask].mean(dim=0)
            counts[c] = mask.sum()

    return centers  # shape [num_classes, feat_dim]