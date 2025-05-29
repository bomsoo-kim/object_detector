# %% [markdown]
# # Reference:
# - [Github] https://github.com/WongKinYiu/yolov7
# - [Github] https://github.com/pjreddie/darknet
# - [Github] https://github.com/ayooshkathuria/pytorch-yolo-v3?tab=readme-ov-file
# - [Github] https://github.com/eriklindernoren/PyTorch-YOLOv3
# - [Github] https://github.com/ayooshkathuria/pytorch-yolo-v3?tab=readme-ov-file
# 
# - [PAPER 2024] A COMPREHENSIVE REVIEW OF YOLO ARCHITECTURES IN COMPUTER VISION: FROM YOLOV1 TO YOLOV8 AND YOLO-NAS: https://arxiv.org/pdf/2304.00501
# - [PAPER 2018] YOLOv3: An Incremental Improvement: https://arxiv.org/abs/1804.02767
# - [PAPER 2016] YOLO9000: Better, Faster, Stronger: https://arxiv.org/pdf/1612.08242
# - [PAPER 2016] You Only Look Once: Unified, Real-Time Object Detection: https://arxiv.org/pdf/1506.02640
# 

# %% [markdown]
# # To Do
# - reproduce performance on COCO dataset
# - train on VOC dataset
# - train on Brad's OCR dataset
# - variable input size: see: inputs = torch.zeros((1, 3, 32, 32)) # error!!!
# 

# %%
import os
print(os.name)

if os.name == 'nt':
    import sys
    # sys.path.append(r'C:\Users\bomso\bomsoo1\python\bradk')
    # sys.path.insert(0, r'C:\Users\bomso\bomsoo1\python\bradk')
    sys.path.insert(0, r'C:\Users\bomso\bomsoo1\python\GitHub\datasets')

else: # if not Windows
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    try: # Google Colab
        from google.colab import drive
        drive.mount('/content/drive')
        # !ls /content/drive/MyDrive/faster_rcnn/datasets_500_CHAR_fontsize20_40

    except Exception as err: # Kaggle kernel
        print(err)
        
        # for dirname, _, filenames in os.walk('/kaggle/input'):
        #     print(dirname)
        #     for filename in filenames:
        #         # print(os.path.join(dirname, filename))
        #         pass

        import sys
        sys.path.append('/kaggle/input/dataset-text-scene')
        sys.path.append('/kaggle/input/dataset-text-scene/bradk/datasets')

# %%
# TURN_ON_COSINE_LR = False # get_consine_lr: max_steps=300*, min_lr=0.01*
TURN_ON_COSINE_LR, DECAY_TYPE, MAX_STEPS = True, ['cosine','linear'][0], 200 # get_consine_lr: min_lr=0.01*
# TURN_ON_COSINE_LR, DECAY_TYPE, MAX_STEPS = True, ['cosine','linear'][0], 300 # get_consine_lr: min_lr=0.01*
# TURN_ON_COSINE_LR, DECAY_TYPE, MAX_STEPS = True, ['cosine','linear'][1], 300 # get_consine_lr: min_lr=0.01*
TURN_ON_BURN_IN = True # original value = True; default value of 'burn-in' = 1000
TURN_ON_LR_DECAY = True # original value = True; threshold = 400000 and then 450000

# BUG_FIX_20250526 = False
BUG_FIX_20250526 = True # proved effective for BRAD_OCR, need to prove for COCO2014 & VOC2007
BRAD_OCR_QUICK_SOLUTION_0520 = True # only for BRAD_OCR dataset

HYPER_PARAMETERS = """
#### original ###############
# # optimizer=adam
# learning_rate=0.0001
# decay=0.0005
# momentum=0.9
# burn_in=1000
#### ADAM (Brad) ###############
optimizer=adam
learning_rate=0.0001
decay=0.00001
# decay=0.0
momentum=0.9
burn_in=1000
#### SGD (Brad) ###############
# optimizer=sgd
# learning_rate=0.001
# decay=0.0005
# momentum=0.9
# burn_in=1000
"""
#--- For debug -----------------------------------------------------------
TURN_ON_PROGRESS_BAR = False # turn on progress bar
# TURN_ON_PROGRESS_BAR = True # turn on progress bar

# %%
if __name__=='__main__':
    import torch

    get_font_filepath = lambda dirpath: [os.path.join(dirpath,f) for f in os.listdir(dirpath)]

    def get_list_words(filepath=None):
        if filepath is None:
            import requests
            response = requests.get('https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt') # https://github.com/dwyl/english-words?tab=readme-ov-file
            list_words = [w.strip() for w in response.text.split('\n') if w.strip()]
        else: # open filepath
            with open(filepath) as file:
                list_words = [w.strip() for w in file.read().split('\n') if w.strip()]
        return list_words


    dataset_inputs = dict(
        # object_type='word', # WORD APPLICATION
        object_type='character', # CHARACTER APPLICATION

        # fix_img_H = None, # WORD APPLICATION / CHARACTER APPLICATION
        # fix_img_H = 120, # CHARACTER APPLICATION
        # fix_img_H = 100, # CHARACTER APPLICATION 2
        fix_img_H = 96, # CHARACTER APPLICATION 2 for YOLOv3

        #--- sample generation from hard drive ------------------------
        dirpaths=[], subdir_imgs='images', subidr_segs='segmentations', filename_annotation='annotation.json',

        #--- sample generation in real time ---------------------------
        # num_real_time_samples=1,
        # num_real_time_samples=500,
        num_real_time_samples=5000,
        # num_real_time_samples=4,

        font_filepath_list=get_font_filepath(
            r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\fonts_321' # for desktop use
            if not torch.cuda.is_available() else
            r'/kaggle/input/dataset-text-scene/fonts_321/fonts_321' # for Kaggle
            ), 

        # generate_text
        # characters = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()_+-={}|[]\:";'<>?,./'''), full
        characters = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+-={}|[]\:";'<>?,./'''), # full except: `
        # characters = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@$/'''), # reduced
        # characters = list('''0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'''), # ONE CHAR

        word_range = range(1,12),
        # word_range = range(1,2), # ONE CHAR

        list_words = [], list_words_prob = 0.0,
        # list_words = get_list_words(filepath= # CHARACTER APPLICATION 3
        #     r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\sample_data\words.txt'
        #     if not torch.cuda.is_available() else
        #     # r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.3,
        #     r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.5,
        #     # r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.7,
        #     # r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.9,

        # num_words_range = range(0,10), # WORD APPLICATION
        # num_words_range = range(5,15), # WORD APPLICATION 2
        num_words_range = range(1,2), # CHARACTER APPLICATION
        # num_lines_range = range(1,20), # WORD APPLICATION
        # num_lines_range = range(10,30), # WORD APPLICATION 2
        num_lines_range = range(1,2), # CHARACTER APPLICATION
        # indent_range = range(0,15), # WORD APPLICATION
        # indent_range = range(0,3), # WORD APPLICATION 2
        indent_range = range(0,1), # CHARACTER APPLICATION

        # generate_OCR_image
        # font_size_range = range(10, 20), # TEST
        # font_size_range = range(20, 40), # TEST
        # font_size_range = range(40, 60), # TEST
        # font_size_range = range(60, 80), # TEST
        font_size_range = range(10, 80),

        # font_size_weights = None,
        font_size_weights = [1/i for i in range(10, 80)],

        line_spacing_range = range(0, 5),

        # img_size_xy=(600, 600), img_size_dx_range=None, img_size_dy_range=None, orig_point_max_ratio_range = 0.5, # WORD APPLICATION
        # img_size_xy=(600, 600), img_size_dx_range=range(0,200), img_size_dy_range=range(0,200), orig_point_max_ratio_range = 0.1, # WORD APPLICATION 2
        # img_size_xy = None, img_size_dx_range=None, img_size_dy_range=None, crop_for_only_characters=False, # CHARACTER APPLICATION
        img_size_xy = None, img_size_dx_range=None, img_size_dy_range=None, crop_for_only_characters=True, # CHARACTER APPLICATION 2

        )

    dataset_inputs_for_train = {**dataset_inputs, 'dirpaths':[
        #     r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\train',
        # ] if device == 'cpu' else [
        #     r'/kaggle/input/dataset-text-scene/datasets_NONE_FONT10_20_BATCH2500/train',
        ]}
    dataset_inputs_for_test = {**dataset_inputs, 'dirpaths':[ # Brad 2024-06-22
            r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\test',
        ] if not torch.cuda.is_available() else [
            # r'/kaggle/input/dataset-text-scene/datasets_ver2_fullexcept_600x600_FONT10_80_BATCH1000/test', # WORD APPLICATION
            # r'/kaggle/input/dataset-text-scene/datasets_ver3_fullexcept_600ax600a_FONT10_80_BATCH1000/test', # WORD APPLICATION 2
            r'/kaggle/input/dataset-text-scene/datasets_ver2_fullexcept_NONE_FONT10_80_BATCH1000/test', # CHARACTER APPLICATION
        ]}


# %%
def get_consine_lr(it=None, warmup_steps=None, max_steps=None, max_lr=None, min_lr=None, decay_type='cosine'):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    if decay_type == 'cosine':
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    else:
        coeff = (it - max_steps) / (warmup_steps - max_steps)
    return min_lr + coeff * (max_lr - min_lr)

# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import math

#     decay_type='cosine'
#     # decay_type='linear'

#     xx = np.array(range(300))
#     yy = np.array([get_consine_lr(x, 10, 200, 0.01, 0.001, decay_type=decay_type) for x in xx])
#     plt.plot(xx,yy, '.-')
#     plt.show()

# %%
import os
import re
import math
import argparse
# import tqdm
from itertools import chain
from typing import List, Tuple
import numpy as np
import time
import platform
import subprocess
import random
import datetime
import glob
import warnings
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision
from torchvision import transforms


# %%
if __name__=='__main__':
    os.system('pip install imgaug')
    os.system('pip install terminaltables')

    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    from terminaltables import AsciiTable

    # from torchsummary import summary

# %% [markdown]
# # Utils

# %% [markdown]
# - utils

# %%
def custom_progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+ 
    """
    https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        elapsed = time.time() - start
        mins, sec = divmod(elapsed, 60) # limited to minutes
        time_str0 = f"{int(mins):02}:{sec:04.1f}"

        remaining = elapsed * (count - j) / j
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:04.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}, elapsed time {time_str0}, estimated wait time {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0 
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

# if __name__=='__main__':
#     # for i in custom_progressbar(range(15), "Computing: ", 40):
#     for i in custom_progressbar(range(15)):
#         time.sleep(0.1) # any code you need

# %%
def provide_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ia.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def to_cpu(tensor):
    return tensor.detach().cpu()


# def load_classes(path):
#     """
#     Loads class labels at 'path'
#     """
#     with open(path, "r") as fp:
#         names = fp.read().splitlines()
#     return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
    for c in custom_progressbar(unique_classes, prefix="Computing AP:") if TURN_ON_PROGRESS_BAR else unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(
        prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
        max_wh = 4096, # Brad: see "c = x[:, 5:6] * max_wh", max_wh = 0 --> not class-wise NMS, while max_wh = 'big number' --> class-wise NMS
        ):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    # max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def print_environment_info():
    """
    Prints infos about the environment and the system.
    This should help when people make issues containg the printout.
    """

    print("Environment information:")

    # Print OS information
    print(f"System: {platform.system()} {platform.release()}")

    # Print poetry package version
    try:
        print(f"Current Version: {subprocess.check_output(['poetry', 'version'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not using the poetry package")

    # Print commit hash if possible
    try:
        print(f"Current Commit Hash: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No git or repo found")

# %% [markdown]
# - loss

# %%
# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py


# def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
def bbox_iou_FOR_LOSS(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9): # Brad
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

# %%
def build_targets(p, targets, model):
    """
        note) only the shape information of predictions is used
        predictions: [(B, A, H//32, W//32, 4 + 1 + num_classes), (B, A, H//16, W//16, 4 + 1 + num_classes), (B, A, H//8, W//8, 4 + 1 + num_classes)]
        targets: (i.e. bb_targets): (nt, 1 + 1 + 4) = (# of boxes, sample_index + label + 4 coordinates)
    """
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    # Brad na = 3 hard-coded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    na, nt = 3, targets.shape[0]  # number of anchors, targets #TODO
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain (cf. tensor([1., 1., 1., 1., 1., 1., 1.]))
    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt) # (na,nt): e.g. [[0,0,...,0], [1,1,...,1], [2,2,...,2]]
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2) # (na,nt,6+1): targets.shape = (nt,6); targets.repeat(na,1,1).shape = (na,nt,6); ai[:, :, None].shape = (na,nt,1)

    for i, yolo_layer in enumerate(model.yolo_layers):
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        anchors = yolo_layer.anchors / yolo_layer.stride # (na,2)
        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain; gain = [1, 1, W//n, H//n, W//n, H//n, 1], i.e. W//n, H//n = feature map size
        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        t = targets * gain
        # Check if we have targets
        if nt:
            # Calculate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors[:, None] # Brad: (na,nt,2) / (na,1,2) # cf) anchors (na,2) --> anchors[:, None] (na,1,2)
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            t = t[j] # Brad: (na,nt,6+1) -> (nt???,6+1) ????????????????????????????????????
        else:
            t = targets[0]

        # Extract image id in batch and class id
        b, c = t[:, :2].long().T # Brad: b = (nt,), c = (nt,)
        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        gxy = t[:, 2:4] # Brad: gxy = (nt,2)
        gwh = t[:, 4:6] # grid wh, # Brad: gwh = (nt,2)
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long() # e.g. gxy = [0.1000, 1.3000, 1.7000, 2.1000] ----> gxy.long() = [0, 1, 1, 2]
        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices # Brad: gi = (nt,), gj = (nt,)

        # Convert anchor indexes to int
        a = t[:, 6].long() # Brad: a = (nt,)
        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # Add correct anchor for each target to the list
        anch.append(anchors[a]) # Brad: (nt,2)
        # Add class for each target to the list
        tcls.append(c)
    # Brad: tcls = [(nt??,)] x layers
    # Brad: tbox = [(nt??,4)] x layers
    # Brad: indices = [[(nt??,), (nt??,), (nt??,), (nt??,)]] x layers
    # Brad: anch = [(nt??,2)] x layers

    return tcls, tbox, indices, anch 

# if __name__=='__main__':
#     na = 3
#     nt = 5
#     targets = torch.arange(nt).view(nt,1).repeat(1,6)
#     print(f'targets = {targets}')
#     ai = torch.arange(na).view(na,1).repeat(1,nt)
#     print(f'ai = {ai}')
#     torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
#     print(targets.repeat(na, 1, 1).shape)
#     print(ai[:, :, None].shape)
#     torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2).shape

# if __name__=='__main__':
#     targets = torch.tensor([[0,0,0,0,0,0], [1,1,1,1,1,1], [2,2,2,2,2,2]])
#     """
#         predictions: [(B, A, H//32, W//32, 4 + 1 + num_classes), (B, A, H//16, W//16, 4 + 1 + num_classes), (B, A, H//8, W//8, 4 + 1 + num_classes)]
#         targets: (i.e. bb_targets): (nt, 1 + 1 + 4) = (# of boxes, sample_index + label + 4 coordinates)
#     """
#     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#     na, nt = 3, targets.shape[0]  # number of anchors, targets #TODO
#     gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#     # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
#     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt) # (na,nt): e.g. [[0,0,...,0], [1,1,...,1], [2,2,...,2]]
#     # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
#     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2) # (na,nt,6+1): targets.shape = (nt,6); targets.repeat(na,1,1).shape = (na,nt,6); ai[:, :, None].shape = (na,nt,1)

# if __name__=='__main__':
#     r = torch.tensor([[1, 1/2],[1, 1/2],[4, 1/3]])[:, None] # Brad: (na,nt,2) / (na,1,2) # cf) anchors (na,2) --> anchors[:, None] (na,1,2)
#     # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
#     j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
#     print(f'>> r = {r}')
#     print(f'>> j = {j}')
#     print(f'>> torch.max(r, 1. / r) = {torch.max(r, 1. / r)}')
#     print(f'>> torch.max(r, 1. / r).max(2) = {torch.max(r, 1. / r).max(2)}')
#     print(f'>> torch.max(r, 1. / r).max(2)[0] = {torch.max(r, 1. / r).max(2)[0]}') # values

# %%
def compute_loss(predictions, targets, model): 
    """
        predictions: [(B, A, H//32, W//32, 4 + 1 + num_classes), (B, A, H//16, W//16, 4 + 1 + num_classes), (B, A, H//8, W//8, 4 + 1 + num_classes)]
        targets: (i.e. bb_targets): (nt, 1 + 1 + 4) = (# of boxes, sample_index + label + 4 coordinates)
    """
    # Check which device was used
    device = targets.device

    # Add placeholder varables for the different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build yolo targets
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets
    # Brad: tcls = [(nt??,)] x layers
    # Brad: tbox = [(nt??,4)] x layers
    # Brad: indices = [[(nt??,), (nt??,), (nt??,), (nt??,)]] x layers
    # Brad: anch = [(nt??,2)] x layers

    # Define different loss functions classification
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    # Calculate losses for each yolo layer
    for layer_index, layer_predictions in enumerate(predictions): # Brad: layer_predictions: (B, A, H//n, W//n, 4 + 1 + num_classes)
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]
        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj # Brad: (B, A, H//n, W//n)
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i] # Brad: (nt, 4 + 1 + num_classes)???

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            # iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            iou = bbox_iou_FOR_LOSS(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            lbox += (1.0 - iou).mean()  # iou loss

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE # Brad: (T, num_classes)

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss # Brad: (B, A, H//n, W//n)

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))

# %% [markdown]
# - transforms

# %%
class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:]) # (cx, cy, w, h) -> (x0, y0, x1, y1)

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage([BoundingBox(*box[1:], label=box[0]) for box in boxes], shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(image=img, bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes # boxes: (Label, cx, cy, w, h)


class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w # (_, x, y, w, h, ) or (_, x, y, x, y, )
        boxes[:, [2, 4]] *= h # (_, x, y, w, h, ) or (_, x, y, x, y, )
        return img, boxes


class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w # (_, x, y, w, h, ) or (_, x, y, x, y, )
        boxes[:, [2, 4]] /= h # (_, x, y, w, h, ) or (_, x, y, x, y, )
        return img, boxes


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img) # (H x W x C) and [0, 255] -> (C x H x W) and [0.0, 1.0]. See: https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes) # tensors are returned without scaling. See: https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html

        return img, bb_targets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes

if __name__=='__main__':
    DEFAULT_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])

# %% [markdown]
# - augmentations

# %%
class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])

if __name__=='__main__':
    AUGMENTATION_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        DefaultAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])

# %% [markdown]
# - datasets

# %%
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size): # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0) # 'nearest' ??? or 'nearest-exact'???
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    LABEL_NAMES = [ # COCO2014
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 
        'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, list_path, data_dir, data_dir_vs_list_file=None, img_size=416, multiscale=True, transform=None):
        self.img_files = [] # list of image files
        self.label_files = [] # list of label files
        for data_dir, list_path in (data_dir_vs_list_file if data_dir_vs_list_file is not None else [(data_dir, list_path)]):
            if isinstance(list_path, list): # Brad
                lines = list_path # Brad
            else:
                with open(list_path, "r") as file:
                    # self.img_files = file.readlines()
                    lines = file.readlines()

            for path_0 in lines:
                if data_dir:
                    path = os.path.join(data_dir, path_0.strip('/\\')) # overwrite path
                else:
                    path = path_0

                self.img_files.append(path)

                image_dir = os.path.dirname(path)
                label_dir = "labels".join(image_dir.rsplit("images", 1))
                assert label_dir != image_dir, f"Image path must contain a folder named 'images'! \n'{image_dir}'"
                label_file = os.path.join(label_dir, os.path.basename(path))
                label_file = os.path.splitext(label_file)[0] + '.txt'
                self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale # variable size of image
        self.min_size = self.img_size - 3 * 32 # applicable only if multiscale = True
        self.max_size = self.img_size + 3 * 32 # applicable only if multiscale = True
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5) # (label, xc, yc, w, h)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets # boxes (B,5) -> bb_targets (B,6), i.e. (B, sample_index + lable + 4)

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

# if __name__=='__main__':
#     import matplotlib.pyplot as plt

#     # list_path = [r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\images\train2014\COCO_train2014_000000000009.jpg']
#     list_path = [r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\images\train2014\COCO_train2014_000000000025.jpg']
#     data_dir = None
#     dataset = ListDataset(list_path, data_dir, img_size=416, multiscale=True, transform=AUGMENTATION_TRANSFORMS)

# #     img = np.array(Image.open(r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO\images\train2014\COCO_train2014_000000000009.jpg'))
# #     print(img.shape) # (480, 640, 3)
# #     print(transforms.ToTensor()(img).shape) # torch.Size([3, 480, 640])

#     img = np.array(Image.open(dataset.img_files[0]))
#     boxes = np.loadtxt(dataset.label_files[0])
#     print(f'img.shape = {img.shape}') # img.shape = (480, 640, 3)
#     print(f'img.min() = {img.min()}') # img.min() = 0
#     print(f'img.max() = {img.max()}') # img.max() = 255
#     print(boxes)
#     print(f'boxes.shape = {boxes.shape}')

#     img2, boxes2 = dataset.transform((img, boxes))
#     print(f'img2.shape = {img2.shape}') # img2.shape = torch.Size([3, 640, 640])
#     print(f'img2.min() = {img2.min()}') # img2.min() = 0.0
#     print(f'img2.max() = {img2.max()}') # img2.max() = 1.0


# %%
class VOC2007ListDataset(Dataset):
    VOC_BBOX_LABEL_NAMES = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor']

    def __init__(self, list_path, data_dir, data_dir_vs_list_file=None, img_size=416, multiscale=True, transform=None, use_difficult=False):
        self.label_names = VOC2007ListDataset.VOC_BBOX_LABEL_NAMES
        self.map_i_to_class_name = {i:c for i,c in enumerate(VOC2007ListDataset.VOC_BBOX_LABEL_NAMES)}
        self.map_class_name_to_i = {c:i for i,c in enumerate(VOC2007ListDataset.VOC_BBOX_LABEL_NAMES)}
        
        self.img_files = []
        self.label_files = []
        for data_dir, list_path in (data_dir_vs_list_file if data_dir_vs_list_file is not None else [(data_dir, list_path)]):
            if isinstance(list_path, list): # Brad
                lines = list_path # Brad
            else:
                with open(list_path, "r") as file:
                    # self.img_files = file.readlines()
                    lines = file.readlines()

            for filename in lines:
                filename = filename.strip()
                if len(filename) == 0: # if empty line
                    continue
                self.img_files.append(os.path.join(data_dir, f'JPEGImages/{filename}.jpg')) # construct the full path
                self.label_files.append(os.path.join(data_dir, f'Annotations/{filename}.xml')) # construct the full path

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale # variable size of image
        self.min_size = self.img_size - 3 * 32 # applicable only if multiscale = True
        self.max_size = self.img_size + 3 * 32 # applicable only if multiscale = True
        self.batch_count = 0
        self.transform = transform
        self.use_difficult = use_difficult

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # # Ignore warning if file is empty
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     boxes = np.loadtxt(label_path).reshape(-1, 5)

            anno = ET.parse(label_path)
            bbox = list()
            label = list()
            difficult = list()
            for obj in anno.findall('object'):
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                if not self.use_difficult and int(obj.find('difficult').text) == 1:
                    continue

                difficult.append(int(obj.find('difficult').text))
                bndbox_anno = obj.find('bndbox')
                # subtract 1 to make pixel indexes 0-based
                xyxy = [
                    int(bndbox_anno.find(tag).text) - 1
                    # for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
                    for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
                # bbox.append(xyxy) # Brad 2025-05-11
                xywh = [0.5*(xyxy[0] + xyxy[2]), 0.5*(xyxy[1] + xyxy[3]), (xyxy[2] - xyxy[0]), (xyxy[3] - xyxy[1])] # Brad 2025-05-11
                bbox.append(xywh) # Brad 2025-05-11
                name = obj.find('name').text.lower().strip()
                label.append(self.label_names.index(name))
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)


            boxes = np.concatenate((label.reshape(-1,1), bbox), axis=1)
            h, w, _ = img.shape
            boxes[:, [1, 3]] /= w # conversion to relative coordinate
            boxes[:, [2, 4]] /= h # conversion to relative coordinate
            #                         
            # print(f'label_path = {label_path}')
            # print(f'bbox = {bbox}')
            # print(f'label = {label}')
            # print(f'boxes = {boxes}')

        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets # boxes (B,5) -> bb_targets (B,6)

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

# if __name__=='__main__':
#     import matplotlib.pyplot as plt

#     list_path = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt'
#     data_dir = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007'
#     dataset = VOC2007ListDataset(list_path, data_dir, img_size=416, multiscale=True, transform=AUGMENTATION_TRANSFORMS)
#     print(dataset[0])


# %%
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import pandas as pd
import cv2
if __name__=='__main__':
    # from bradk.datasets.advanced_texts import generate_random_sample
    from advanced_texts import generate_random_sample

def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def pre_process_image_bboxes(img, bboxes=None, fix_img_H=None):
    if fix_img_H: # resize image and bbox
        H, W, C = img.shape
        img = cv2.resize(img.astype(np.float32), (math.ceil(W * fix_img_H / H), math.ceil(fix_img_H)), interpolation=(cv2.INTER_AREA if (fix_img_H / H) < 1 else cv2.INTER_LINEAR))
        if bboxes is not None:
            o_H, o_W, _ = img.shape
            bboxes = resize_bbox(bboxes, (H, W), (o_H, o_W)) # Brad: also resize bbox according to the resized (or preprocessed) image

    return img, bboxes

# class AdvancedTextsDataset_YXYX(torch.utils.data.Dataset):
class AdvancedTextsDataset_xywh(torch.utils.data.Dataset):
    def __init__(
            self,
            object_type=['word','character'][0],
            fix_img_H = None,
            #--- sample generation from hard drive ------------------------
            dirpaths=[], 
            subdir_imgs='images', subidr_segs='segmentations', filename_annotation='annotation.json',

            #--- sample generation in real time ---------------------------
            num_real_time_samples=1, 
            font_filepath_list=["arial.ttf"],
            # generate_text
            characters = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()_+-={}|[]\\:";'<>?,./'''),
            word_range = range(1,12),
            list_words = [],
            list_words_prob = 0, # 0 <= list_words_prob <=1
            num_words_range = range(0,10),
            num_lines_range = range(1,20),
            indent_range = range(0,15),
            # generate_OCR_image
            font_size_range = range(10, 100),
            font_size_weights = None,
            line_spacing_range = range(0, 5),
            orig_point_max_ratio_range = 0.5,
            angle_range=range(-5,6),
            img_size_xy = (600, 600),
            img_size_dx_range=None,
            img_size_dy_range=None,
            crop_for_only_characters=False, # effective only if object_type=='character'
            # table parameters
            prob_draw_top_line = 0.3, # [0, 1]
            prob_draw_bottom_line = 0.3, # [0, 1]
            prob_draw_left_line = 0.3, # [0, 1]
            prob_draw_right_line = 0.3, # [0, 1]
            prob_draw_inner_line_yoffset = 0.3, # [0, 1]

            max_iter = 5,
            ):

        self.object_type = object_type
        self.fix_img_H = fix_img_H

        self.dirpaths = dirpaths
        #--- samples saved in hard drive -------------------------
        if dirpaths:
            print(f'#--- samples saved in hard drive -------------------------')
            self.subdir_imgs = subdir_imgs
            self.subidr_segs = subidr_segs

            self.characters = None # overwrite, if any, to reset

            data = []
            for d in self.dirpaths:
                info = json.load(open(os.path.join(d, filename_annotation), 'r'))

                for datum in info['annotation']:
                    datum['dirpath'] = d # add dirpath
                    data.append(datum)

                if self.characters is None:
                    self.characters = list(info['characters'])
                else:
                    if self.characters != list(info['characters']):
                        raise(Exception(f"Brad error: the sets of 'characters' are not compatible: {self.characters} VS. {list(info['characters'])}"))

            self.annotation = pd.DataFrame(data)

        #--- samples randomly generated in real time -------------
        else:
            print(f'#--- samples randomly generated in real time -------------')
            self.num_real_time_samples = num_real_time_samples

            self.font_filepath_list = font_filepath_list

            self.characters = characters
            self.word_range = word_range
            self.list_words = list_words
            self.list_words_prob = list_words_prob
            self.num_words_range = num_words_range
            self.num_lines_range = num_lines_range
            self.indent_range = indent_range
            # generate_OCR_image
            self.font_size_range = font_size_range
            self.font_size_weights = font_size_weights
            self.line_spacing_range = line_spacing_range
            self.orig_point_max_ratio_range = orig_point_max_ratio_range
            self.angle_range = angle_range
            self.img_size_xy = img_size_xy
            self.img_size_dx_range = img_size_dx_range
            self.img_size_dy_range = img_size_dy_range
            # table parameters
            self.prob_draw_top_line = prob_draw_top_line
            self.prob_draw_bottom_line = prob_draw_bottom_line
            self.prob_draw_left_line = prob_draw_left_line
            self.prob_draw_right_line = prob_draw_right_line
            self.prob_draw_inner_line_yoffset = prob_draw_inner_line_yoffset

            self.max_iter = max_iter

        #-------------------------------------------------------------
        self.crop_for_only_characters = crop_for_only_characters

        if self.object_type == 'word':
            self.label_names = ['text']
            self.obj_to_idx = lambda x: 0
        elif self.object_type == 'character':
            self.label_names = self.characters
            self.obj_to_idx = lambda x, c_to_i={c:i for i,c in enumerate(self.characters)}: c_to_i[x]
        else:
            raise(Exception(f"Brad error: no such object type, '', ..."))

        return

    def __len__(self):
        if self.dirpaths:
            return len(self.annotation) # samples saved in hard drive
        else:
            return self.num_real_time_samples # samples randomly generated in real time

    def __getitem__(self, index):
        #--- samples saved in hard drive -------------------------
        if self.dirpaths:
            dirpath = self.annotation.loc[index,'dirpath']
            filename_image = self.annotation.loc[index,'filename_image']
            # filename_segmentation = self.annotation.loc[index,'filename_segmentation']

            image = cv2.imread(os.path.join(dirpath, self.subdir_imgs, filename_image))

            if self.object_type == 'word':
                bboxes = self.annotation.loc[index,'bboxes_word']
                labels = [self.obj_to_idx(x) for x in self.annotation.loc[index,'labels_word']]
            else: # character
                bboxes = self.annotation.loc[index,'bboxes_char']
                labels = [self.obj_to_idx(x) for x in self.annotation.loc[index,'labels_char']]
                bboxes_word = self.annotation.loc[index,'bboxes_word']

            # rboxes = eval(self.annotation.loc[index,'rboxes'])
            # angles = eval(self.annotation.loc[index,'angles'])
            # segmentation = cv2.imread(os.path.join(dirpath, self.subidr_segs, filename_segmentation))[:,:,0]

        #--- samples randomly generated in real time -------------
        else:
            for i_try in range(self.max_iter + 1):
                try:
                    filepath = random.choice(self.font_filepath_list)

                    if self.img_size_xy is not None:
                        dx = 0 if self.img_size_dx_range is None else random.choice(self.img_size_dx_range)
                        dy = 0 if self.img_size_dy_range is None else random.choice(self.img_size_dy_range)
                        x0, y0 = self.img_size_xy
                        img_size_xy_NEW = (x0 + dx, y0 + dy)
                    else:
                        img_size_xy_NEW = self.img_size_xy

                    s = generate_random_sample(
                        # generate_text
                        characters = self.characters,
                        word_range = self.word_range,
                        list_words = self.list_words,
                        list_words_prob = self.list_words_prob, # 0 <= list_words_prob <=1
                        list_words_cap_types = ['as-is','lower','upper','title'],
                        list_words_cap_types_weights = None, # ex) [1,1,1,1]. Note: len(list_words_cap_types) == len(list_words_cap_types_weights)
                        space_range = [1,2,3],
                        space_weights = [3,2,1],
                        num_words_range = self.num_words_range,
                        num_lines_range = self.num_lines_range,
                        indent_range = self.indent_range,

                        # generate_OCR_image
                        font_size_range = self.font_size_range,
                        font_size_weights = self.font_size_weights,
                        line_spacing_range = self.line_spacing_range,
                        orig_point_max_ratio_range = self.orig_point_max_ratio_range,
                        font_color_max_range = 120,
                        background_color_max_range = 120,
                        background_color_std=(20,20,20),
                        angle_range=self.angle_range,                        
                        filepath=filepath,
                        # img_size_xy = self.img_size_xy,
                        img_size_xy = img_size_xy_NEW,
                        min_img_size_xy=(16, 16),
                        overlap_ratio_threshold=0.7, draw_bboxes=False,

                        # table parameters
                        prob_draw_top_line = self.prob_draw_top_line, # [0, 1]
                        prob_draw_bottom_line = self.prob_draw_bottom_line, # [0, 1]
                        prob_draw_left_line = self.prob_draw_left_line, # [0, 1]
                        prob_draw_right_line = self.prob_draw_right_line, # [0, 1]
                        prob_draw_inner_line_yoffset = self.prob_draw_inner_line_yoffset, # [0, 1]
                        table_line_width_range = range(1,4),
                        table_line_color_max_range = 120,
                        table_top_line_margin_range = range(1,6),
                        table_bottom_line_margin_range = range(1,6),
                        table_left_line_margin_range = range(1,6),
                        table_right_line_margin_range = range(1,6),
                        table_inner_line_yoffset_range = range(-3,4),
                        table_outer_space_margin = 2,

                        debug=False,
                        # debug=True,
                    )
                    assert len(s['labels_char']) > 0, 'Brad error: generated is an image with no text...'
                    break

                except Exception as err:
                    if i_try >= self.max_iter:
                        raise(err)
                    print(f'{err}')

            image = np.array(s['image'])

            if self.object_type == 'word':
                bboxes = s['bboxes_word']
                labels = [self.obj_to_idx(x) for x in s['labels_word']]
            else: # character
                bboxes = s['bboxes_char']
                labels = [self.obj_to_idx(x) for x in s['labels_char']]
                bboxes_word = s['bboxes_word']

        #--- data clean for faster rnn model ---------------------------
        if self.object_type=='character' and self.crop_for_only_characters:
            xb0, yb0, xb1, yb1 = bboxes_word[0] # there is only one word for object_type=='character'
            H, W, _ = image.shape
            xn0 = max(0, math.floor(xb0))
            yn0 = max(0, math.floor(yb0))
            xn1 = min(W, math.ceil(xb1))
            yn1 = min(H, math.ceil(yb1))

            image = image[yn0:yn1, xn0:xn1, :] # crop image
            bboxes = np.array([[y0-yn0, x0-xn0, y1-yn0, x1-xn0] for x0, y0, x1, y1 in bboxes]) # YXYX, i.e. [y0, x0, y1, x1], representation
        else:
            bboxes = np.array([[y0, x0, y1, x1] for x0, y0, x1, y1 in bboxes]) # YXYX, i.e. [y0, x0, y1, x1], representation
        labels = np.array(labels)

        img = image # (H, W, C)
        img, bboxes = pre_process_image_bboxes(img, bboxes=bboxes, fix_img_H=self.fix_img_H)

        difficult = np.array([0]*len(labels)) # Brad 2024-06-22

        return img, bboxes, labels, difficult # Brad 2024-06-22

    def collate_fn(self, batch):
        MAX_H, MAX_W = 0, 0
        for img, bboxes, labels, difficult in batch:
            H, W, C = img.shape
            MAX_H = max(MAX_H, H)
            MAX_W = max(MAX_W, W)
        
        if BRAD_OCR_QUICK_SOLUTION_0520:
            n, r = divmod(MAX_W, 32)
            MAX_W = (n + 1*(r>0))*32

        paths, imgs, bb_targets = [], [], []
        for i, (img, bboxes, labels, difficult) in enumerate(batch):
            paths.append(None) # dummy action, because paths will not be useed

            #--- image --------------------------
            H, W, C = img.shape
            img_padded = np.pad(img, ((0, MAX_H - H), (0, MAX_W - W), (0,0)), 'constant', constant_values=0) # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            imgs.append(img_padded)

            #--- bboxes: YXYX -> XYWH and normalize with H and W -------------
            y0 = bboxes[:,0]
            x0 = bboxes[:,1]
            y1 = bboxes[:,2]
            x1 = bboxes[:,3]

            bboxes_xywh = np.zeros((bboxes.shape[0], bboxes.shape[1] + 2), dtype=np.float32) # (B, sample_index + lable + 4)
            bboxes_xywh[:,0] = i
            bboxes_xywh[:,1] = labels
            bboxes_xywh[:,2] = 0.5*(x0 + x1) / MAX_W
            bboxes_xywh[:,3] = 0.5*(y0 + y1) / MAX_H
            bboxes_xywh[:,4] = (x1 - x0) / MAX_W
            bboxes_xywh[:,5] = (y1 - y0) / MAX_H

            bb_targets.append(bboxes_xywh)

        imgs = torch.tensor(np.stack(imgs)).permute(0,3,1,2) / 255.0 # transpose: (B, H, W, C) -> (B, C, H, W) and then normalize [0, 255] --> [0.0, 1.0]
        bb_targets = torch.tensor(np.concatenate(bb_targets, axis=0))

        return paths, imgs, bb_targets # bb_targets (B,6), i.e. (B, sample_index + lable + 4)


# %% [markdown]
# - parse_config

# %%
# def parse_model_config(path, debug=False):
def parse_model_config(path_or_lines, debug=False): # Brad
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    if isinstance(path_or_lines, list): # Brad
        lines = path_or_lines # Brad
    else:
        # file = open(path, 'r') 
        file = open(path_or_lines, 'r') # Brad
        lines = file.read().split('\n')
    if debug:
        print(lines)
    lines = [x for x in lines if x and not x.startswith('#')] # exclude an empty or a comment line
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    if debug:
        print(lines)
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip() # exclude '[' and ']'
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

# if __name__=='__main__':
#     # path = r'C:\Users\bomso\bomsoo1\python\GitHub\object_detector\config\yolov3.cfg'
#     # module_defs = parse_model_config(path, debug=True)
#     module_defs = parse_model_config(CONFIG_LINES_FOR_YOLO_V3)
#     for i, module_def in enumerate(module_defs):
#         print(f'[{i}/{len(module_defs)}] {module_def}')

# %%
def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

# %% [markdown]
# - logger

# %%
class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

# %% [markdown]
# # Models

# %%
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/config/yolov3.cfg
CONFIG_LINES_FOR_YOLO_V3 = f"""
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=16
subdivisions=1
width=416
height=416
channels=3
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

#### original ###############
# # optimizer=adam
# learning_rate=0.0001
# decay=0.0005
# momentum=0.9
# burn_in=1000

{HYPER_PARAMETERS}

###############################

max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
""".split('\n')

# print(CONFIG_LINES_FOR_YOLO_V3)

# %%
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode: str = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

# %%
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2) # Brad: why not just anchors = torch.tensor(anchors) ??
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor: # Brad: input - (batch_size, num_anchors*(5 + num_classes), hh, ww)
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2) # Height: _, _, H, _ = x.shape
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # Brad: (batch_size, num_anchors, hh, ww, 5 + num_classes)

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            if self.new_coords:
                x[..., 0:2] = (x[..., 0:2] + self.grid) * stride  # xy
                x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid) # wh
            else:
                x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
                x[..., 4:] = x[..., 4:].sigmoid() # conf, cls
            x = x.view(bs, -1, self.no) # Brad: x(bs, 3, 20, 20, 5 + num_classes) --> x(bs, -1, 5 + num_classes)

        return x # Brad: output - (batch_size, num_anchors, hh, ww, 5 + num_classes), if training else, (batch_size, num_anchors * hh * ww, 5 + num_classes)

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    
# if __name__=='__main__':
#     anchors = [[1,2],[3,4],[5,6]]
#     print(torch.tensor(list(chain(*anchors))).float().view(-1, 2))

# %%
# def create_modules(module_defs: List[dict]) -> Tuple[dict, nn.ModuleList]: # BRAD_20250520
def create_modules(module_defs, n_classes=None): # BRAD_20250520
    """
    Constructs module list of layer blocks from module configuration in module_defs

    :param module_defs: List of dictionaries with module definitions
    :return: Hyperparameters and pytorch module list
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        # 'batch': int(hyperparams['batch']), # BRAD_20250520
        # 'subdivisions': int(hyperparams['subdivisions']), # BRAD_20250520
        # 'width': int(hyperparams['width']), # BRAD_20250520
        # 'height': int(hyperparams['height']), # BRAD_20250520
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    # assert hyperparams["height"] == hyperparams["width"], "Height and width should be equal! Non square images are padded with zeros." # BRAD_20250520
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            # filters = int(module_def["filters"]) # BRAD_20250520
            filters = int(module_def["filters"]) if (n_classes is None or int(module_def["filters"]) != 255) else 3*(5 + n_classes) # BRAD_20250520
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", nn.Mish())
            elif module_def["activation"] == "logistic":
                modules.add_module(f"sigmoid_{module_i}", nn.Sigmoid())
            elif module_def["activation"] == "swish":
                modules.add_module(f"swish_{module_i}", nn.SiLU())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            # example)
            # mask = 3,4,5
            # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            # classes=80
            # num=9
            # jitter=.3
            # ignore_thresh = .7
            # truth_thresh = 1
            # random=1            
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"]) if n_classes is None else n_classes # BRAD_20250520
            new_coords = bool(module_def.get("new_coords", False))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, new_coords)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

# if __name__=='__main__':
#     module_defs = parse_model_config(CONFIG_LINES_FOR_YOLO_V3) # Brad
#     hyperparams, module_list = create_modules(module_defs)    


# %%
class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    # def __init__(self, config_path): # BRAD_20250520
    def __init__(self, config_path, n_classes=None): # BRAD_20250520
        super(Darknet, self).__init__()
        if config_path is not None: # Brad
            self.module_defs = parse_model_config(config_path)
            print(f'> model configuration is successfully loaded from: {config_path}')
        else:
            self.module_defs = parse_model_config(CONFIG_LINES_FOR_YOLO_V3) # Brad
            print(f'> No model config file is specified. So, the default configuration for yolo v3 is applied. check "CONFIG_LINES_FOR_YOLO_V3" inside the code.')
        # self.hyperparams, self.module_list = create_modules(self.module_defs) # BRAD_20250520
        self.hyperparams, self.module_list = create_modules(self.module_defs, n_classes=n_classes) # BRAD_20250520
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # print(f'BRAD_DEBUG: i = {i}, module_def["type"] = {module_def["type"]}, x.shape = {x.shape}')
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        # Brad: output - (batch_size, num_anchors, hh, ww, 5 + num_classes), if training else, (batch_size, num_anchors * hh * ww, 5 + num_classes)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

# %%
# https://www.geeksforgeeks.org/darknet-53/

class CBLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBLBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            CBLBlock(channels, channels // 2, kernel_size=1, stride=1, padding=0),
            CBLBlock(channels // 2, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class CBLx5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBLx5, self).__init__()
        self.block = nn.Sequential(
            CBLBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            CBLBlock(out_channels, 2*out_channels, kernel_size=3, stride=1, padding=1),
            CBLBlock(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            CBLBlock(out_channels, 2*out_channels, kernel_size=3, stride=1, padding=1),
            CBLBlock(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.block(x)


class CBLConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBLConv, self).__init__()
        self.block = nn.Sequential(
            CBLBlock(in_channels, 2*in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2*in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True), # activatoin linear ???? # (num_classes + 5)*num_channels = 255
        )

    def forward(self, x):
        return self.block(x)


class Darknet53_YOLOv3(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(Darknet53_YOLOv3, self).__init__()

        self.cbl_init = CBLBlock(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.res1   = nn.Sequential(CBLBlock(32, 64, kernel_size=3, stride=2, padding=1), ResidualBlock(64))
        self.res2   = nn.Sequential(CBLBlock(64, 128, kernel_size=3, stride=2, padding=1), *[ResidualBlock(128) for _ in range(2)])
        self.res8_1 = nn.Sequential(CBLBlock(128, 256, kernel_size=3, stride=2, padding=1), *[ResidualBlock(256) for _ in range(8)])
        self.res8_2 = nn.Sequential(CBLBlock(256, 512, kernel_size=3, stride=2, padding=1), *[ResidualBlock(512) for _ in range(8)])
        self.res4   = nn.Sequential(CBLBlock(512, 1024, kernel_size=3, stride=2, padding=1), *[ResidualBlock(1024) for _ in range(4)])

        # no SPP block implemented

        self.cbl5_1 = CBLx5(1024, 512)
        self.cbl_conv_1 = CBLConv(512, (5 + num_classes)*num_channels)
        self.yolo_1 = YOLOLayer(anchors=[(116,90), (156,198), (373,326)], num_classes=num_classes, new_coords=False)

        self.cbl_upsample_2 = nn.Sequential(CBLBlock(512, 256, kernel_size=1, stride=1, padding=0), Upsample(scale_factor=2, mode="nearest"))
        self.cbl5_2 = CBLx5(256 + 512, 256)
        self.cbl_conv_2 = CBLConv(256, (5 + num_classes)*num_channels)
        self.yolo_2 = YOLOLayer(anchors=[(30,61),  (62,45),  (59,119)], num_classes=num_classes, new_coords=False)

        self.cbl_upsample_3 = nn.Sequential(CBLBlock(256, 128, kernel_size=1, stride=1, padding=0), Upsample(scale_factor=2, mode="nearest"))
        self.cbl5_3 = CBLx5(128 + 256, 128)
        self.cbl_conv_3 = CBLConv(128, (5 + num_classes)*num_channels)
        self.yolo_3 = YOLOLayer(anchors=[(10,13),  (16,30),  (33,23)], num_classes=num_classes, new_coords=False)

    def forward(self, x):
        img_size = x.size(2)

        x = self.cbl_init(x)
        x = self.res1(x)
        x = self.res2(x)
        r81 = self.res8_1(x)
        r82 = self.res8_2(r81)
        r4 = self.res4(r82)

        l1 = self.cbl5_1(r4)
        c1 = self.cbl_conv_1(l1)
        y1 = self.yolo_1(c1, img_size)

        u2 = self.cbl_upsample_2(l1)
        l2 = self.cbl5_2(torch.cat([u2, r82], 1))
        c2 = self.cbl_conv_2(l2)
        y2 = self.yolo_2(c2, img_size)

        u3 = self.cbl_upsample_3(l2)
        l3 = self.cbl5_3(torch.cat([u3, r81], 1))
        c3 = self.cbl_conv_3(l3)
        y3 = self.yolo_3(c3, img_size)

        yolo_outputs = [y1, y2, y3]
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    def load_darknet_weights(self, weights_path):
        with open(weights_path, "rb") as f: # Open the weights file
            header = np.fromfile(f, dtype=np.int32, count=5) # First five are header values
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i, (name, module) in enumerate(model.named_modules()):
            if ptr >= len(weights):
                break

            if isinstance(module, CBLBlock):
                conv_layer = module.conv
                if any(isinstance(c, torch.nn.BatchNorm2d) for c in module.children()): # Load BN bias, weights, running mean and running variance
                    bn_layer = module.bn
                    num_b = bn_layer.bias.numel()  # Number of biases

                    _ = bn_layer.bias.data.copy_(torch.from_numpy(weights[ptr: (ptr + num_b)]).view_as(bn_layer.bias)) # Bias
                    ptr += num_b
                    _ = bn_layer.weight.data.copy_(torch.from_numpy(weights[ptr: (ptr + num_b)]).view_as(bn_layer.weight)) # Weight
                    ptr += num_b
                    _ = bn_layer.running_mean.data.copy_(torch.from_numpy(weights[ptr: (ptr + num_b)]).view_as(bn_layer.running_mean)) # Running Mean
                    ptr += num_b
                    _ = bn_layer.running_var.data.copy_(torch.from_numpy(weights[ptr: (ptr + num_b)]).view_as(bn_layer.running_var)) # Running Var
                    ptr += num_b

                else: # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    _ = conv_layer.bias.data.copy_(torch.from_numpy(weights[ptr: (ptr + num_b)]).view_as(conv_layer.bias))
                    ptr += num_b

                # Load conv. weights
                num_w = conv_layer.weight.numel()
                _ = conv_layer.weight.data.copy_(torch.from_numpy(weights[ptr: (ptr + num_w)]).view_as(conv_layer.weight))
                ptr += num_w
 
        if ptr != len(weights):
            raise(Exception("Brad error: the model parameters don't match the loaded weights..."))
        else:
            print(f'Succeesfully loaded weights from the file = {weights_path}')
        return


# if __name__ == "__main__":
#     model = Darknet53_YOLOv3(num_classes=80, num_channels=3)
#     # model.load_darknet_weights(r'D:\codes\OCR\models\darknet53.conv.74')
#     model.load_darknet_weights(r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\darknet53.conv.74')
#     # Sample input tensor with batch size of 1 and image size 416x416
#     x = torch.randn(1, 3, 416, 416)
#     output = model(x)
#     print(output[0].shape)  # torch.Size([1, 3, 13, 13, 85])
#     print(output[1].shape)  # torch.Size([1, 3, 26, 26, 85])
#     print(output[2].shape)  # torch.Size([1, 3, 52, 52, 85])


# %%
# def load_model(model_path, weights_path=None): # BRAD_20250520
def load_model(model_path, n_classes=None, weights_path=None): # BRAD_20250520
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device for inference
    # model = Darknet(model_path).to(device) # BRAD_20250520
    model = Darknet(model_path, n_classes=n_classes).to(device) # BRAD_20250520

    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model

# if __name__=='__main__':
#     model = load_model(None, None)
#     # print(model)
#     # inputs = torch.zeros((1, 3, 416, 416))
#     # inputs = torch.zeros((1, 3, 64, 64))
#     # inputs = torch.zeros((1, 3, 60, 60))
#     # inputs = torch.zeros((1, 3, 64, 416))
#     # inputs = torch.zeros((2, 3, 32, 32))
#     inputs = torch.zeros((2, 3, 64, 20))
#     # inputs = torch.zeros((2, 3, 64, 32)) # error!!!
#     # inputs = torch.zeros((1, 3, 32, 32)) # error --> https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/3
#     print(f'inputs.shape = {inputs.shape}')
#     #-------------------------------------------------------------
#     outputs = model(inputs)
#     for out in outputs:
#         print(f'out.shape = {out.shape}')
#     #-------------------------------------------------------------
#     # x = inputs
#     # for i in range(len(model.module_list)):
#     #     y = model.module_list[i](x)
#     #     print(f'[{i}] y.shape = {y.shape}')
#     #     x = y
#     #-------------------------------------------------------------
#     # inputs.shape = torch.Size([1, 3, 416, 416])
#     # out.shape = torch.Size([1, 3, 13, 13, 85]) # 416 / 32 = 13
#     # out.shape = torch.Size([1, 3, 26, 26, 85]) # 416 / 16 = 26
#     # out.shape = torch.Size([1, 3, 52, 52, 85]) # 416 / 8 = 52

# %% [markdown]
# # Test

# %%
# def evaluate_model_file(model_path, data_dir, data_dir_vs_list_file, weights_path, img_path, class_names, batch_size=8, img_size=416,
def evaluate_model_file(model_path, data_dir, data_dir_vs_list_file, weights_path, img_path, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, dataset_name='COCO2014', verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader, class_names = _create_validation_data_loader(img_path, data_dir, data_dir_vs_list_file, batch_size, img_size, n_cpu, dataset_name=dataset_name)
    # model = load_model(model_path, weights_path) # BRAD_20250520
    model = load_model(model_path, n_classes=len(class_names), weights_path=weights_path) # BRAD_20250520
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
    for _, imgs, targets in custom_progressbar(dataloader, prefix="Validating:") if TURN_ON_PROGRESS_BAR else dataloader:
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        if BUG_FIX_20250526:
            _, _,  H, W = imgs.shape
            targets[:, 2] *= W # x
            targets[:, 3] *= H # y
            targets[:, 4] *= W # w
            targets[:, 5] *= H # h
        else:
            targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs) # Brad: output - (batch_size, num_anchors, hh, ww, 5 + num_classes), if training else, (batch_size, num_anchors * hh * ww, 5 + num_classes)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


def _create_validation_data_loader(img_path, data_dir, data_dir_vs_list_file, batch_size, img_size, n_cpu, dataset_name='COCO2014'):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    if dataset_name =='COCO2014':
        dataset = ListDataset(img_path, data_dir, data_dir_vs_list_file, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
        class_names = ListDataset.LABEL_NAMES
    elif dataset_name =='VOC2007':
        # dataset = VOC2007ListDataset(img_path, data_dir, data_dir_vs_list_file, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS, use_difficult=True)
        dataset = VOC2007ListDataset(img_path, data_dir, data_dir_vs_list_file, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS, use_difficult=False) # Brad update: 2025-05-13
        class_names = VOC2007ListDataset.VOC_BBOX_LABEL_NAMES
    elif dataset_name =='BRAD_OCR':
        dataset = AdvancedTextsDataset_xywh(**dataset_inputs_for_test) # only for test
        class_names = dataset.label_names
    else:
        raise(Exception(f'Brad error: no such dataset name: "{dataset_name}" ...'))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)
    return dataloader, class_names


def run_test(command_args):
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    # parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-m", "--model", type=str, help="Path to model definition file (.cfg)") # Brad
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    # parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("--dataset_name", type=str, default="COCO2014", help="COCO2014, VOC2007, etc.") # Brad
    parser.add_argument("--data_dir", type=str, default="data/coco", help="Path to the root directory where 'images' and 'annotations' sub-folders exist") # Brad
    parser.add_argument("--valid", type=str, default="data/coco/5k.part", help="Path to the file containg the list of validation image files") # Brad
    parser.add_argument("--test_dir_vs_list_file", type=str, action="append", nargs="+") # data/coco data/coco/trainvalno5k.part
    # parser.add_argument("--names", type=str, default="data/coco.names", help="Path to the file containg the list of all the class names") # Brad
    parser.add_argument("--names", type=str, help="Path to the file containg the list of all the class names") # Brad
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    # args = parser.parse_args()
    args = parser.parse_args(command_args) # Brad: for jupyter notebook testing    
    print(f"Command line arguments: {args}")

    # if args.dataset_name =='COCO2014':
    #     # class_names = load_classes(args.names)  # List of class names
    #     class_names = ListDataset.LABEL_NAMES
    # elif args.dataset_name =='VOC2007':
    #     class_names = VOC2007ListDataset.VOC_BBOX_LABEL_NAMES

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model, args.data_dir, args.test_dir_vs_list_file, 
        args.weights,
        args.valid, # Path to file containing all images for validation
        # class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        dataset_name=args.dataset_name,
        verbose=True)
    return

# %%
# if __name__ == "__main__":
#     if torch.cuda.is_available():
#         command_args = re.findall('[^\s=]+', fr"""
#             --dataset_name COCO2014
#             --data_dir /kaggle/input/coco-2014-dataset-for-yolov3/coco2014
#             --valid /kaggle/input/coco-2014-dataset-for-yolov3/coco2014/5k.part
#             --names /kaggle/input/coco-2014-dataset-for-yolov3/coco2014/coco.names
#             --weights /kaggle/input/datasets-yolov3/yolov3.weights
#             --n_cpu 4
#             """)
#     else:
#         command_args = re.findall('[^\s=]+', fr"""
#             --dataset_name COCO2014
#             --data_dir C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO
#             --valid C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO\5k.part
#             --names C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO\coco.names
#             --weights C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO\yolov3.weights
#             --n_cpu 0
#             """)
#     print(command_args)

#     history = run_test(command_args)

# %% [markdown]
# # Detect

# %%
def _create_data_loader_inference(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader

# %%
def detect(model, dataloader, output_path, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    # for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
    for (img_paths, input_imgs) in custom_progressbar(dataloader, prefix="Detecting:") if TURN_ON_PROGRESS_BAR else dataloader:
        
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs) # Brad: output - (batch_size, num_anchors, hh, ww, 5 + num_classes), if training else, (batch_size, num_anchors * hh * ww, 5 + num_classes)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs


# %%
def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """

    # Iterate through images and save plot of detections
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(image_path, detections, img_size, output_path, classes)


# %%
def detect_directory(model_path, weights_path, img_path, classes, output_path,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    dataloader = _create_data_loader_inference(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    img_detections, imgs = detect(model, dataloader, output_path, conf_thres, nms_thres)
    _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes)

    print(f"---- Detections were saved to: '{output_path}' ----")


# %%
def run_detect(command_args):
    print_environment_info()
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--images", type=str, default="data/samples", help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    # args = parser.parse_args()
    args = parser.parse_args(command_args) # Brad: for jupyter notebook testing
    print(f"Command line arguments: {args}")
    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    detect_directory(
        args.model,
        args.weights,
        args.images,
        classes,
        args.output,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres)

# if __name__=='__main__':
#     if torch.cuda.is_available():
#         command_args = re.findall('[^\s=]+', fr"""
#             --dataset_name BRAD_OCR
#             --pretrained_weights /kaggle/input/datasets-yolov3/yolov3.weights
#             --epochs 10
#             --checkpoint_interval 10
#             --n_cpu 4
#             """)
#     else:
#         command_args = re.findall('[^\s=]+', fr"""
#             --dataset_name BRAD_OCR
#             --pretrained_weights C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\yolov3.weights
#             --epochs 1
#             --batch_size 16
#             --n_cpu 0
#             """)

#     ##########################################################################################
#     print(command_args)

#     history = run_detect(command_args)


# %%
def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])((image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img) # Brad: output - (batch_size, num_anchors, hh, ww, 5 + num_classes), if training else, (batch_size, num_anchors * hh * ww, 5 + num_classes)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()

# %%
def batch_extract_bboxes_from_BRAD_OCR_images(model, class_names, imgs, fix_img_H, x0y0s=None, conf_thres=0.5, iou_thres=0.5, max_wh=0, draw_bboxes=False):
    if x0y0s is None:
        x0y0hs = [(0, 0, img.shape[0]) for img in imgs]
    else:
        x0y0hs = [(x0, y0, img.shape[0]) for img, (x0, y0) in zip(imgs, x0y0s)]

    MAX_W = 0
    imgs_resized = []
    for img in imgs:
        img_resized, _ = pre_process_image_bboxes(img, bboxes=None, fix_img_H=fix_img_H)
        imgs_resized.append(img_resized)

        H, W, _ = img_resized.shape
        assert H == fix_img_H
        MAX_W = max(MAX_W, W)

    #--- re-adjust MAX_W, to make it multiples of 32 -----------------------------
    n, r = divmod(MAX_W, 32)
    MAX_W = (n + 1*(r>0))*32

    #--- make the width uniform ----------------------------------------------
    imgs_padded = []
    for img_resized in imgs_resized:
        hh, ww, _ = img_resized.shape
        img_padded = np.pad(img_resized, ((0, 0), (0, MAX_W - ww), (0,0)), 'constant', constant_values=0) # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        imgs_padded.append(img_padded)

    #--- tensor input -------------------------------------------------------
    input_img = torch.tensor(np.stack(imgs_padded)).permute(0,3,1,2) / 255.0 # transpose: (B, H, W, C) -> (B, C, H, W) and then normalize [0, 255] --> [0.0, 1.0]

    #--- predict ------------------------------------
    # Get detections
    model.eval()  # Set model to evaluation mode
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    detections = model(input_img) # Brad: output - (batch_size, num_anchors, hh, ww, 5 + num_classes), if training else, (batch_size, num_anchors * hh * ww, 5 + num_classes)
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    detections = non_max_suppression(detections, conf_thres=conf_thres, iou_thres=iou_thres, max_wh=max_wh)
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    #--- rescale bboxes -----------------------------
    detections_rescaled = []
    for detection, (x0, y0, orig_H) in zip(detections, x0y0hs):
        detection_rescaled = []
        scale = orig_H / fix_img_H
        for x1, y1, x2, y2, conf, cls_pred in detection.numpy():
            detection_rescaled.append((x0 + x1*scale, y0 + y1*scale, x0 + x2*scale, y0 + y2*scale, conf, class_names[int(cls_pred)]))
        detection_rescaled = sorted(detection_rescaled, key=lambda x: x[0])
        detections_rescaled.append(detection_rescaled)

    #--- plot bboxes and labels -------------------------------------------
    if draw_bboxes:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        for img, detection_rescaled, (x0, y0, orig_H) in zip(imgs, detections_rescaled, x0y0hs):
            fig, ax = plt.subplots(1)
            plt.imshow(img)

            labels = []
            for x1, y1, x2, y2, conf, label in detection_rescaled:
                x0_ = x1 - x0
                y0_ = y1 - y0
                width = x2 - x1
                height = y2 - y1

                ax.add_patch(patches.Rectangle((x0_, y0_), width, height, linewidth=1, edgecolor='r', facecolor='none')) # Add the rectangle to the axes
                ax.text(x0_, y0_, f'{label}:{conf:.2f}', style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0}, fontsize=10)

                labels.append(label)

            plt.show() # Show the plot
            print(f"{''.join(labels)} ::: {detection_rescaled}")

    return detections_rescaled

# if __name__=='__main__':
#     from PIL import Image

#     # dataset_name, class_names, pretrained_weights = 'COCO2014', ListDataset.LABEL_NAMES, r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\yolov3.weights'
#     # dataset_name, class_names, pretrained_weights = 'VOC2007', VOC2007ListDataset.VOC_BBOX_LABEL_NAMES, r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\yolov3.weights'
#     dataset_name, class_names, pretrained_weights = 'BRAD_OCR', list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+-={}|[]\:";'<>?,./'''), r'C:\Users\bomso\Downloads\yolov3_ckpt_20250528_1913_100.pth'
#     model = load_model(None, n_classes=len(class_names), weights_path=pretrained_weights) # BRAD_20250520

#     # print(detections)

#     # img = np.array(Image.open(r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\images\val2014\COCO_val2014_000000000192.jpg'))
#     # img = np.array(Image.open(r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\images\val2014\COCO_val2014_000000000387.jpg'))

#     imgs = [np.array(Image.open(r'C:\Users\bomso\Downloads\Copy of0002.png')) for _ in range(100)]
#     x0y0s = [(0,0) for _ in range(len(imgs))]
#     fix_img_H = 32
#     detections_rescaled = batch_extract_bboxes_from_BRAD_OCR_images(model, class_names, imgs, fix_img_H, x0y0s=x0y0s)
#     print(detections_rescaled)


# %% [markdown]
# # Train

# %%
def _create_data_loader(img_path, data_dir, data_dir_vs_list_file, batch_size, img_size, n_cpu, multiscale_training=False, dataset_name='COCO2014'):
    
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    if dataset_name =='COCO2014':
        dataset = ListDataset(img_path, data_dir, data_dir_vs_list_file, img_size=img_size, multiscale=multiscale_training, transform=AUGMENTATION_TRANSFORMS)
        class_names = ListDataset.LABEL_NAMES
    elif dataset_name =='VOC2007':
        dataset = VOC2007ListDataset(img_path, data_dir, data_dir_vs_list_file, img_size=img_size, multiscale=multiscale_training, transform=AUGMENTATION_TRANSFORMS, use_difficult=False)
        class_names = VOC2007ListDataset.VOC_BBOX_LABEL_NAMES
    elif dataset_name =='BRAD_OCR':
        dataset = AdvancedTextsDataset_xywh(**dataset_inputs_for_train)
        class_names = dataset.label_names
    else:
        raise(Exception(f'Brad error: no such dataset name: "{dataset_name}" ...'))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu, pin_memory=True, collate_fn=dataset.collate_fn, worker_init_fn=worker_seed_set)
    return dataloader, class_names


# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_w_bbox(imgs, targets, class_names):
    _, _, H, W = imgs.shape
    imgs_ = imgs.permute(0,2,3,1)

    for i in range(len(imgs_)):
        img = imgs_.numpy()[i]

        bboxes = targets[targets[:,0] == i]

        # Create a figure and axes
        fig, ax = plt.subplots(1)
        plt.imshow(img)
        # x, y = 0,0
        # width, height = 10, 20
        

        for _, ilabel, x, y, w, h in bboxes:
            width = w*W
            height = h*H
            x0 = (x - 0.5*w)*W
            y0 = (y - 0.5*h)*H

            ax.add_patch(patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')) # Add the rectangle to the axes
            ax.text(x0, y0, class_names[int(ilabel)], style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0}, fontsize=10)

        plt.show() # Show the plot

# if __name__=='__main__':
#     # dataset_name = 'COCO2014'
#     # data_dir = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014'
#     # train = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\trainvalno5k.part'

#     # dataset_name = 'VOC2007'
#     # data_dir = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007'
#     # train = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt'

#     dataset_name = 'BRAD_OCR'
#     data_dir = None
#     train = None

#     dataloader, class_names = _create_data_loader(train, data_dir, None, 2, 416, 0, False, dataset_name=dataset_name)
    
#     for batch_i, (_, imgs, targets) in enumerate(dataloader):
#         break

#     display_image_w_bbox(imgs, targets, class_names)

# %%
def run_train(command_args, history={}):
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    # parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-m", "--model", type=str, help="Path to model definition file (.cfg)") # Brad
    # parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("--dataset_name", type=str, default="COCO2014", help="COCO2014, VOC2007, etc.") # Brad
    parser.add_argument("--data_dir", type=str, default="data/coco", help="Path to the root directory where 'images' and 'annotations' sub-folders exist") # Brad
    parser.add_argument("--train", type=str, default="data/coco/trainvalno5k.part", help="Path to the file containg the list of training image files") # Brad
    parser.add_argument("--valid", type=str, default="data/coco/5k.part", help="Path to the file containg the list of validation image files") # Brad
    parser.add_argument("--train_dir_vs_list_file", type=str, action="append", nargs="+") # data/coco data/coco/trainvalno5k.part
    parser.add_argument("--valid_dir_vs_list_file", type=str, action="append", nargs="+") # data/coco data/coco/5k.part
    # parser.add_argument("--names", type=str, default="data/coco.names", help="Path to the file containg the list of all the class names") # Brad
    parser.add_argument("--names", type=str, help="Path to the file containg the list of all the class names") # Brad
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--img_size", type=int, default=416, help="only for COCO and VOC datasets") # BRAD_20250520
    parser.add_argument("--batch_size", type=int, default=16) # BRAD_20250520
    parser.add_argument("--subdivisions", type=int, default=1) # BRAD_20250520
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    # args = parser.parse_args()
    args = parser.parse_args(command_args) # Brad: for jupyter notebook testing
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # misc --------------------------------
    # if args.dataset_name =='COCO2014':
    #     # class_names = load_classes(args.names) # Brad
    #     class_names = ListDataset.LABEL_NAMES
    # elif args.dataset_name =='VOC2007':
    #     class_names = VOC2007ListDataset.VOC_BBOX_LABEL_NAMES

    # print(f'>> len(class_names) = {len(class_names)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'>> device = {device}')

    # #################
    # Create Dataloader
    # #################

    # mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions'] # BRAD_20250520
    mini_batch_size = args.batch_size // args.subdivisions # BRAD_20250520

    # Load training dataloader
    dataloader, class_names = _create_data_loader(
        args.train, args.data_dir, args.train_dir_vs_list_file,
        mini_batch_size,
        # model.hyperparams['height'], # BRAD_20250520
        args.img_size, # BRAD_20250520
        args.n_cpu,
        args.multiscale_training, dataset_name=args.dataset_name)

    print(f'>> len(class_names) = {len(class_names)}')

    # Load validation dataloader
    validation_dataloader, _ = _create_validation_data_loader(
        args.valid, args.data_dir, args.valid_dir_vs_list_file,
        mini_batch_size,
        # model.hyperparams['height'], # BRAD_20250520
        args.img_size, # BRAD_20250520
        args.n_cpu, dataset_name=args.dataset_name)

    # ############
    # Create model
    # ############

    print(f'>> pretrained_weights = {args.pretrained_weights}')

    # model = load_model(args.model, args.pretrained_weights) # BRAD_20250520
    model = load_model(args.model, n_classes=len(class_names), weights_path=args.pretrained_weights) # BRAD_20250520

    # # Print model
    # if args.verbose:
    #     summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if model.hyperparams['optimizer'] in [None, "adam"]:
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif model.hyperparams['optimizer'] == "sgd":
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        # for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        for batch_i, (_, imgs, targets) in enumerate(custom_progressbar(dataloader, prefix=f"Training Epoch {epoch}:")) if TURN_ON_PROGRESS_BAR else enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            # Brad: (batch_size, num_anchors, hh, ww, 5 + num_classes)
            outputs = model(imgs) # outputs: [(B, A, H//32, W//32, 4 + 1 + num_classes), (B, A, H//16, W//16, 4 + 1 + num_classes), (B, A, H//8, W//8, 4 + 1 + num_classes)]

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            # if batches_done % model.hyperparams['subdivisions'] == 0: # BRAD_20250520
            if batches_done % args.subdivisions == 0: # BRAD_20250520
                if TURN_ON_COSINE_LR:
                    NNN = len(dataloader)
                    lr = get_consine_lr(
                        NNN * len(history) + batch_i, 
                        warmup_steps=model.hyperparams['burn_in'], 
                        max_steps=MAX_STEPS * NNN, 
                        max_lr=model.hyperparams['learning_rate'], 
                        min_lr=model.hyperparams['learning_rate'] * 0.01,
                        decay_type=DECAY_TYPE,
                        )
                else:
                    # Adapt learning rate
                    # Get learning rate defined in cfg
                    lr = model.hyperparams['learning_rate']
                    if batches_done < model.hyperparams['burn_in']: # default value of 'burn-in' = 1000
                        # Burn in
                        if TURN_ON_BURN_IN:
                            lr *= (batches_done / model.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in model.hyperparams['lr_steps']:
                            if batches_done > threshold: # Brad: threshold = 400000 and then 450000
                                if TURN_ON_LR_DECAY:
                                    lr *= value # Brad: the default of 'value' is set to 0.1, so it means lr = 0.1 * lr
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{time.strftime('%Y%m%d_%H%M')}_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                # img_size=model.hyperparams['height'], # BRAD_20250520
                img_size=args.img_size, # BRAD_20250520
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                history.append({
                    'epoch':epoch,
                    'lr':optimizer.param_groups[0]['lr'],
                    'mAP':AP.mean(),
                    'precision':precision.mean(),
                    'recall':recall.mean(),
                    'f1':f1.mean(),
                })
                evaluation_metrics = [
                    ("validation/precision", history[-1]['precision']),
                    ("validation/recall", history[-1]['recall']),
                    ("validation/mAP", history[-1]['mAP']),
                    ("validation/f1", history[-1]['f1'])]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                json.dump(history, open('history.json', 'w'))
    return history

# %% [markdown]
# - load model

# %%
if __name__=='__main__':
    if False:
        # d = r'/kaggle/input/pjt-faster-rcnn-20240622/checkpoints'
        # all_trained = sorted([os.path.join(d,f) for f in os.listdir(d)])
        # print(all_trained)
        # filepath_trained = all_trained[-1]
        # print(f'filepath_trained = {filepath_trained}')

        # pretrained_weights = filepath_trained
        pretrained_weights = r''

        history = json.load(open(r'/kaggle/input/pjt-faster-rcnn-20240622/history.json', 'r'))
        seed = -1
    else:
        pretrained_weights = None
        history = []
        seed = 12345

# %% [markdown]
# - run train

# %%
if __name__ == "__main__":
    ### COCO2014 #############################################################################
    # if torch.cuda.is_available():
    #     command_args = re.findall('[^\s=]+', fr"""
    #         --dataset_name COCO2014
    #         --data_dir /kaggle/input/coco-2014-dataset-for-yolov3/coco2014
    #         --train /kaggle/input/coco-2014-dataset-for-yolov3/coco2014/trainvalno5k.part
    #         --valid /kaggle/input/coco-2014-dataset-for-yolov3/coco2014/5k.part
    #         --names /kaggle/input/coco-2014-dataset-for-yolov3/coco2014/coco.names
    #         --pretrained_weights {'/kaggle/input/datasets-yolov3/darknet53.conv.74' if pretrained_weights is None else pretrained_weights}

    #         --epochs 10
    #         --checkpoint_interval 10

    #         --n_cpu 4
    #         --seed {seed}
    #         """)
    # else:
    #     command_args = re.findall('[^\s=]+', fr"""
    #         --dataset_name COCO2014
    #         --data_dir C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014
    #         --train C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\trainvalno5k.part
    #         --valid C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\5k.part
    #         --names C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\coco.names
    #         --pretrained_weights C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\darknet53.conv.74
    #         --epochs 1
    #         --batch_size 16
    #         --n_cpu 0
    #         --seed 12345
    #         """)

    ### VOC2007 #############################################################################
    # if torch.cuda.is_available():
    #     command_args = re.findall('[^\s=]+', fr"""
    #         --dataset_name VOC2007
    #         --train_dir_vs_list_file /kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007
    #                                     /kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
    #         --train_dir_vs_list_file /kaggle/input/pascal-voc-2012/VOC2012
    #                                     /kaggle/input/pascal-voc-2012/VOC2012/ImageSets/Main/trainval.txt
    #         --valid_dir_vs_list_file /kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007
    #                                     /kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt
    #         --multiscale_training
    #         --pretrained_weights {'/kaggle/input/datasets-yolov3/darknet53.conv.74' if pretrained_weights is None else pretrained_weights}

    #         --epochs 60
    #         --checkpoint_interval 60

    #         --n_cpu 4
    #         --seed {seed}
    #         """)
    # else:
    #     command_args = re.findall('[^\s=]+', fr"""
    #         --dataset_name VOC2007
    #         --data_dir C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007
    #         --train C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt
    #         --valid C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007\ImageSets\Main\test.txt
    #         --pretrained_weights C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\darknet53.conv.74
    #         --epochs 1
    #         --batch_size 16
    #         --n_cpu 0
    #         --seed 12345
    #         """)

    ### BRAD_OCR #############################################################################
    if torch.cuda.is_available():
        command_args = re.findall('[^\s=]+', fr"""
            --dataset_name BRAD_OCR
            --pretrained_weights {'/kaggle/input/datasets-yolov3/darknet53.conv.74' if pretrained_weights is None else pretrained_weights}
            --epochs 10
            --checkpoint_interval 10
            --n_cpu 4
            --seed {seed}
            """)
    else:
        command_args = re.findall('[^\s=]+', fr"""
            --dataset_name BRAD_OCR
            --pretrained_weights C:\Users\bomso\bomsoo1\python\_pytorch\data\COCO2014\darknet53.conv.74
            --epochs 1
            --batch_size 16
            --n_cpu 0
            --seed 12345
            """)

    ##########################################################################################
    print(command_args)

    history = run_train(command_args, history=history)

# %% [markdown]
# - Plot History

# %%
if __name__=='__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option('display.max_rows', 500)

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    df[['mAP']].plot(ax=axes[0], figsize=(12,4), logy=False)
    df[['precision','recall','f1']].plot(ax=axes[1], figsize=(12,4), logy=False)
    plt.show()

    display(df)

# %%



