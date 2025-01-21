from torch import nn
from medpy.metric.binary import assd, dc
from datetime import datetime

import scipy.io as scio

import torch.backends.cudnn as cudnn
import os
import cv2
from PIL import Image
from torch.nn import functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
import scipy.ndimage as nd

BATCHSIZE = 32
data_size     = [256, 256, 1]
label_size    = [256, 256, 1]
NUMCLASS      = 5

def eval_supervised(testfile_list, model):
    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()

    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []

    for idx_file, fid in enumerate(testfile_list):
        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0']
        label = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        tmp_pred = np.zeros(label.shape)
        frame_list = [kk for kk in range(data.shape[2])]
        pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                if 