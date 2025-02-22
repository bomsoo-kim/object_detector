# %% [markdown]
# # Focus Items:
# - Feature Pyramid Network
# - YOLO: anchor selection algorithm
# - YOLO: loss function - MSE
# ---------------------------------
# - training dataset: dictionary words --> proved effective!
# 

# %% [markdown]
# # Reference:
# 
# - [Github] https://github.com/chenyuntc/simple-faster-rcnn-pytorch
# - [Github] https://github.com/ShaoqingRen/faster_rcnn
# - [Github] https://github.com/longcw/faster_rcnn_pytorch
# - [Github] https://github.com/jwyang/faster-rcnn.pytorch
# - [Github] https://github.com/trzy/FasterRCNN
# - [Github] https://github.com/tryolabs/object-detection-workshop/blob/master/Implementing%20Faster%20R-CNN.ipynb
# - Guide to build Faster RCNN in PyTorch: https://medium.com/@fractal.ai/guide-to-build-faster-rcnn-in-pytorch-42d47cb0ecd3
# - [Youtube] How FasterRCNN works and step-by-step PyTorch implementation: https://www.youtube.com/watch?v=4yOcsWg-7g8
# - [PAPER 2015] Very Deep Convolutional Networks for Large-Scale Image Recognition: https://arxiv.org/abs/1409.1556
# - [PAPER 2015] Deep Residual Learning for Image Recognition: https://arxiv.org/pdf/1512.03385
# - [PAPER 2016] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks: https://arxiv.org/abs/1506.01497
# - [PAPER 2017] Feature Pyramid Networks for Object Detection: https://arxiv.org/pdf/1612.03144
# - [PAPER 2018] Mask R-CNN: https://arxiv.org/abs/1703.06870
# - https://www.youtube.com/watch?v=a5yDWeSoudE
# 
# - An Improved Faster R-CNN for Small Object Detection: https://ieeexplore.ieee.org/abstract/document/8786135
# - A closer look: Small object detection in faster R-CNN: https://ieeexplore.ieee.org/document/8019550
# 
# - What is Mask R-CNN? The Ultimate Guide: https://blog.roboflow.com/mask-rcnn/
# - Everything About Mask R-CNN: A Beginner’s Guide: https://viso.ai/deep-learning/mask-r-cnn/
# - Instance Segmentation: https://tjmachinelearning.com/lectures/1718/instance/instance.pdf
# - What is instance segmentation?: https://www.ibm.com/topics/instance-segmentation
# - CSE5194: ResNet and ResNeXt: https://drago1234.github.io/about_me/pdf/CS5194_ResNet_v2.0.pdf
# - ResNeXt: A New Paradigm in Image Processing: https://medium.com/@atakanerdogan305/resnext-a-new-paradigm-in-image-processing-ee40425aea1f
# - https://www.youtube.com/watch?v=fqVMa03iPVE
# 
# - [PAPER 2024] A COMPREHENSIVE REVIEW OF YOLO ARCHITECTURES IN COMPUTER VISION: FROM YOLOV1 TO YOLOV8 AND YOLO-NAS: https://arxiv.org/pdf/2304.00501
# - [PAPER 2018] YOLOv3: An Incremental Improvement: https://pjreddie.com/media/files/papers/YOLOv3.pdf
# - [PAPER 2016] YOLO9000: Better, Faster, Stronger: https://arxiv.org/pdf/1612.08242
# - [PAPER 2016] You Only Look Once: Unified, Real-Time Object Detection: https://arxiv.org/pdf/1506.02640
# 

# %%
TURN_ON_YOLOv2 = False # BRAD: 2025-01-20
# TURN_ON_YOLOv2 = True # BRAD: 2025-01-20

PATCH_FEATURE_PYRAMID = False

BATCH_SIZE_TRAIN, BATCH_SIZE_TEST = 1, 1 # multi-batch
# BATCH_SIZE_TRAIN, BATCH_SIZE_TEST = 2, 1 # multi-batch

#--- confirmed patches ---------------------------------------------------
PATCH_REMOVE_INSIDE_INDEX_LIMIT = False # AnchorTargetCreator
PATCH_REMOVE_INSIDE_INDEX_LIMIT_2 = False # ProposalCreator
PATCH_REMOVE_INSIDE_INDEX_LIMIT_3 = False # predict > _suppress

#--- For debug -----------------------------------------------------------
TURN_ON_PROGRESS_BAR = False # turn on progress bar
# TURN_ON_PROGRESS_BAR = True # turn on progress bar


# %%
import os
print(os.name)

if os.name == 'nt':
    import sys
    # sys.path.append(r'C:\Users\bomso\bomsoo1\python\bradk')
    sys.path.insert(0, r'C:\Users\bomso\bomsoo1\python\bradk')

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

# %%
# if os.name != 'nt':
    # !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    # !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    # !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

    # !tar xf VOCtrainval_06-Nov-2007.tar
    # !tar xf VOCtest_06-Nov-2007.tar
    # !tar xf VOCdevkit_08-Jun-2007.tar

    # os.system('ls')

# %%
import os
import time
import re
import statistics
import math
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections import defaultdict # https://www.geeksforgeeks.org/defaultdict-in-python/
import json
import datetime
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

import torch
from torch import nn
from torch.nn import Flatten
from torch.nn import functional as F
import torchvision
from torchvision.ops import nms
from torchvision.ops import RoIPool, RoIAlign
from torchvision.models import vgg16 # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
from torchvision.models import resnet34, resnet50, resnet101, resnet152
from torchvision.models import resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
from torch.utils.data import DataLoader

# %% [markdown]
# # Configuration

# %%
from pprint import pprint

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/dataset/PASCAL2007/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3. # https://github.com/rbgirshick/py-faster-rcnn/issues/89
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'  # visdom env

    # preset
    data = 'voc'

    # training
    epoch = 14

    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs, verbose=True):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        if verbose:
            print('======user config========')
            pprint(self._state_dict())
            print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()

if __name__=='__main__':
    test = Config()
    pprint(Config.__dict__)

# %% [markdown]
# # Metrics Function

# %%
class ConfusionMeter(): # https://tnt.readthedocs.io/en/latest/_modules/torchnet/meter/confusionmeter.html
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], 'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, 'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), 'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, 'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), 'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), 'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), 'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

# if __name__=='__main__': # Brad
#     cm = ConfusionMeter(2)
#     cm.add(torch.tensor([0,1,1,1]), torch.tensor([0,1,1,1]))
#     print(cm.value())
#     cm.add(torch.tensor([0,0,1,1]), torch.tensor([0,1,1,0]))
#     print(cm.value())
#     print(type(cm.value()))

# %% [markdown]
# # Util

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
def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img)

    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        return img[..., np.newaxis] # reshape (H, W) -> (H, W, 1)

    else:
        return img # (H, W, C)

# if __name__=='__main__': # Brad
#     filepath = r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007\JPEGImages\000001.jpg'
#     img = read_image(filepath, dtype=np.float32, color=True)

#     print(f'img.shape = {img.shape}')
#     print(f'img.dtype = {img.dtype}')

    # plt.imshow(img)

# %%
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

# if __name__=='__main__': # Brad
#     bbox = np.array([[0,0,1,2]], dtype=np.float32) # [y_min, x_min, y_max, x_max]
#     print(f'bbox = {bbox}')
#     in_size = (1, 2)
#     out_size = (2, 3)

#     bbox_resized = resize_bbox(bbox, in_size, out_size)
#     print(f'bbox_resized = {bbox_resized}')

# %%
def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


# %%
def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


# %%
def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.

    This method is mainly used together with image cropping.
    This method translates the coordinates of bounding boxes like
    :func:`data.util.translate_bbox`. In addition,
    this function truncates the bounding boxes to fit within the cropped area.
    If a bounding box does not overlap with the cropped area,
    this bounding box will be removed.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_slice (slice): The slice of y axis.
        x_slice (slice): The slice of x axis.
        allow_outside_center (bool): If this argument is :obj:`False`,
            bounding boxes whose centers are outside of the cropped area
            are removed. The default value is :obj:`True`.
        return_param (bool): If :obj:`True`, this function returns
            indices of kept bounding boxes.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`, returns an array :obj:`bbox`.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`bbox, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **index** (*numpy.ndarray*): An array holding indices of used \
            bounding boxes.

    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


# %%
def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


# %%
def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[::-1, :, :] # (H, _, _)
    if x_flip:
        img = img[:, ::-1, :] # (_, W, _)

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


# %% [markdown]
# # VOC Dataset

# %%
class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False):
        VOC_BBOX_LABEL_NAMES = (
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
            'tvmonitor')

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
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
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        # difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
        difficult = np.array(difficult, dtype=bool).astype(np.uint8)  # PyTorch don't support np.bool #### Brad

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True) # Brad: read image from path, and then convert from uint8 to float32 and transpose into (C, H, W)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example

# %%
class VOCDataset:
    def __init__(self, opt, split=['trainval','test'][0], use_difficult=False):
        self.opt = opt
        self.split = split
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
        self.min_size = opt.min_size
        self.max_size = opt.max_size

    @staticmethod
    def transform_image(img, bbox, label, min_size=600, max_size=1000, augment_image=True):
        H, W, _ = img.shape
        #-- resize image --------------------------------------------------
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = cv2.resize(img.astype(np.float32), (math.ceil(W * scale), math.ceil(H * scale)), interpolation=(cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
        #-----------------------------------------------------
        o_H, o_W, _ = img.shape
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W)) # Brad: also resize bbox according to the resized (or preprocessed) image

        # horizontally flip
        if augment_image:
            img, params = random_flip(img, x_random=True, return_param=True)
            bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        if self.split == 'trainval':
            img, bbox, label = self.transform_image(ori_img, bbox, label, min_size=self.min_size, max_size=self.max_size, augment_image=True) # preprocess + resize_bbox + random flip (augmentation)
        elif self.split == 'test':
            img, bbox, label = self.transform_image(ori_img, bbox, label, min_size=self.min_size, max_size=self.max_size, augment_image=False)

        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), difficult.copy()

    def __len__(self):
        return len(self.db)

# %% [markdown]
# # Array Tool

# %%
"""
tools to convert specified type
"""

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

def to_tensor(data, cuda=torch.cuda.is_available()): # Brad
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

def to_scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()

# if __name__=='__main__': # Brad
#     a = np.array([[1,2,3],[4,5,6]])
#     b = torch.tensor(a)
#     print(to_numpy(b))
#     print(to_tensor(a))
#     print(to_scalar(np.array([[1]])))
#     print(to_scalar(torch.tensor([[1]])))
#     print(to_numpy(b))

# %% [markdown]
# # Bbox Tools

# %%
def loc2bbox(src_bbox, loc): # Brad: src_bbox = "reference bbox"
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """
    if src_bbox.shape[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_height = src_bbox[:, 2] - src_bbox[:, 0] # Brad: y_max - y_min
    src_width = src_bbox[:, 3] - src_bbox[:, 1] # Brad: x_max - x_min
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0]
    dx = loc[:, 1]
    dh = loc[:, 2]
    dw = loc[:, 3]

    ctr_y = dy * src_height + src_ctr_y
    ctr_x = dx * src_width + src_ctr_x
    h = torch.exp(dh) * src_height
    w = torch.exp(dw) * src_width

    if torch.cuda.is_available(): # Brad
        dst_bbox = torch.empty(loc.shape, dtype = loc.dtype, device = "cuda")
    else: # Brad
        dst_bbox = torch.empty(loc.shape, dtype = loc.dtype, device = "cpu") # Brad
    dst_bbox[:, 0] = ctr_y - 0.5 * h # y_min
    dst_bbox[:, 1] = ctr_x - 0.5 * w # x_min
    dst_bbox[:, 2] = ctr_y + 0.5 * h # y_max
    dst_bbox[:, 3] = ctr_x + 0.5 * w # x_max

    return dst_bbox

# if __name__=='__main__': # Brad
#     src_bbox = np.array([[0, 0, 1, 1]], dtype=np.float32) # y_min, x_min, y_max, x_max
#     loc = np.array([[0.5, 0.5, 0, 0]], dtype=np.float32) # dy, dx, dh ,dw
#     # loc = np.array([[0, 0, np.log(2), np.log(2)]], dtype=np.float32) # dy, dx, dh ,dw

#     dst_bbox = loc2bbox(src_bbox, loc)

#     print(f'src_bbox = {src_bbox}')
#     print(f'loc = {loc}')
#     print(f'dst_bbox = {dst_bbox}')

# %%
def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def bbox2loc_torch(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = torch.tensor(torch.finfo(height.dtype).eps)
    height = torch.maximum(height, eps)
    width = torch.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = torch.log(base_height / height)
    dw = torch.log(base_width / width)

    # loc = torch.vstack((dy, dx, dh, dw)).T # transpose
    if torch.cuda.is_available(): # Brad
      loc = torch.empty((len(dy), 4), dtype = torch.float32, device = "cuda") # (N,4)
    else:
      loc = torch.empty((len(dy), 4), dtype = torch.float32, device = "cpu") # (N,4)
    loc[:, 0] = dy
    loc[:, 1] = dx
    loc[:, 2] = dh
    loc[:, 3] = dw

    return loc

# if __name__=='__main__': # Brad
#     src_bbox = np.array([[0, 0, 1, 1]], dtype=np.float32) # y_min, x_min, y_max, x_max
#     dst_bbox = np.array([[0.5, 0.5, 1.5, 1.5]], dtype=np.float32) # y_min, x_min, y_max, x_max

#     loc = bbox2loc(src_bbox, dst_bbox)

#     print(f'src_bbox = {src_bbox}')
#     print(f'dst_bbox = {dst_bbox}')
#     print(f'loc = {loc}')

# if __name__=='__main__': # Brad
#     src_bbox = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32) # y_min, x_min, y_max, x_max
#     dst_bbox = torch.tensor([[0.5, 0.5, 1.5, 1.5]], dtype=torch.float32) # y_min, x_min, y_max, x_max

#     loc = bbox2loc_torch(src_bbox, dst_bbox)

#     print(f'src_bbox = {src_bbox}')
#     print(f'dst_bbox = {dst_bbox}')
#     print(f'loc = {loc}')

# %%
def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2]) # top left
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:]) # bottom right

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox_iou_torch(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2]) # top left # (N,1,2) and (M,2) -> (N,M,2) indicating top-left corners of box pairs
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:]) # bottom right # "" bottom-right corners ""

    _mask_ = torch.all(tl < br, axis = 2) # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
    area_i = torch.prod(br - tl, dim = 2) * _mask_ # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1) # (N,) indicating areas of boxes1
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1) # (M,) indicating areas of boxes2
    return area_i / (area_a[:, None] + area_b - area_i) # # (N,1) + (M,) - (N,M) = (N,M), union areas of both boxes
    # epsilon = 1e-7
    # return area_i / (area_a[:, None] + area_b - area_i + epsilon)

if __name__=='__main__': # Brad
    bbox_a = np.array([[0, 0, 1, 1], [0.5,0.5,1.5,1.5]], dtype=np.float32) # y_min, x_min, y_max, x_max
    bbox_b = np.array([[0.5,0.5,1.5,1.5], [1,1,2,2]], dtype=np.float32) # y_min, x_min, y_max, x_max

    ious = bbox_iou(bbox_a, bbox_b)

    print(f'ious = {ious}')

# %%
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None, debug=False):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
    function.
    The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

    For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
    the width and the height of the base window will be stretched by :math:`8`.
    For modifying the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in :obj:`anchor_scales` and the original area of the reference window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    cx = cy = base_size / 2.
    
    if num_offsets is not None: # overwrite 'x_offsets' and 'y_offsets'
        sub_base_size = base_size / num_offsets
        x_offsets = [(0.5 + i)*sub_base_size - cx for i in range(num_offsets)]
        y_offsets = [(0.5 + i)*sub_base_size - cy for i in range(num_offsets)]
        if debug:
            print(f'x_offsets = {x_offsets}')
            print(f'y_offsets = {y_offsets}')

    anchor_base = np.zeros((len(ratios) * len(anchor_scales) * len(y_offsets) * len(x_offsets), 4), dtype=np.float32) # initialize
    cnt = 0
    for ratio in ratios:
        for scale in anchor_scales:
            for dy in y_offsets:
                for dx in x_offsets:
                    h = base_size * scale * np.sqrt(ratio)
                    w = base_size * scale * np.sqrt(1. / ratio)

                    anchor_base[cnt, 0] = cy + dy - h / 2. # y_min
                    anchor_base[cnt, 1] = cx + dx - w / 2. # x_min
                    anchor_base[cnt, 2] = cy + dy + h / 2. # y_max
                    anchor_base[cnt, 3] = cx + dx + w / 2. # x_max
                    cnt += 1
    return anchor_base

# if __name__=='__main__': # Brad
#     # anchor_base = generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32])
#     anchor_base = generate_anchor_base(base_size=16, ratios=[1], anchor_scales=[1], num_offsets=7)
#     # anchor_base = generate_anchor_base(base_size=16, ratios=[1], anchor_scales=[1, 2])
#     # anchor_base = generate_anchor_base(base_size=16, ratios=[1], anchor_scales=[0], y_offsets=[-16/3*1, 0, 16/3*1], x_offsets=[-16/3*1, 0, 16/3*1])
#     # anchor_base = generate_anchor_base(
#     #     base_size=16, ratios=[1], anchor_scales=[0], 
#     #     y_offsets=[-16/5*2, -16/5*1, 0, 16/5*1, 16/5*2], 
#     #     x_offsets=[-16/5*2, -16/5*1, 0, 16/5*1, 16/5*2])
#     print(anchor_base)
#     # Output:
#     # anchor_base = 
#     # array([[ -37.254833,  -82.50967 ,   53.254833,   98.50967 ],
#     #     [ -82.50967 , -173.01933 ,   98.50967 ,  189.01933 ],
#     #     [-173.01933 , -354.03867 ,  189.01933 ,  370.03867 ],
#     #     [ -56.      ,  -56.      ,   72.      ,   72.      ],
#     #     [-120.      , -120.      ,  136.      ,  136.      ],
#     #     [-248.      , -248.      ,  264.      ,  264.      ],
#     #     [ -82.50967 ,  -37.254833,   98.50967 ,   53.254833],
#     #     [-173.01933 ,  -82.50967 ,  189.01933 ,   98.50967 ],
#     #     [-354.03867 , -173.01933 ,  370.03867 ,  189.01933 ]],
#     #     dtype=float32)


# %% [markdown]
# # Eval Tool

# %%
def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.

    """

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            # selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            seen = np.zeros(gt_bbox_l.shape[0], dtype=bool) # Brad
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        # if not selec[gt_idx]:
                        if not seen[gt_idx]: # Brad
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    # selec[gt_idx] = True
                    seen[gt_idx] = True # Brad: check seen
                else:
                    match[l].append(0)

    for iter_ in (pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0, the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec

# if __name__=='__main__': # Brad
#     pred_bboxes = [
#         np.array([[0.1, 0.1, 1.1, 1.1], [3.9, 3.9, 4.9, 4.9]]),
#     ]
#     pred_labels = [
#         np.array([0, 0]),
#     ]
#     pred_scores = [
#         np.array([1.0, 1.0])
#     ]
#     gt_bboxes = [
#         np.array([[0,0,1,1], [2,2,3,3], [4,4,5,5]]),
#     ]
#     gt_labels = [
#         np.array([0, 0, 0]),
#     ]

#     prec, rec = calc_detection_voc_prec_rec(
#         pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
#         gt_difficults=None,
#         iou_thresh=0.5)
#     print(f'prec = {prec}')
#     print(f'rec = {rec}')

# %%
# https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
# https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


# %%
def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted bounding boxes obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **AP** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **mAP** (*float*): The average of Average Precisions over classes.

    """

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    if use_07_metric is not None:
        ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
        return {'AP': ap, 'mAP': np.nanmean(ap)}
    else:
        ap_07 = calc_detection_voc_ap(prec, rec, use_07_metric=True)
        ap = calc_detection_voc_ap(prec, rec, use_07_metric=False)
        return {'AP_07': ap_07, 'mAP_07': np.nanmean(ap_07), 'AP': ap, 'mAP': np.nanmean(ap)}


# %% [markdown]
# # Vis Tool

# %%
def vis_bbox(img, bbox, label=None, score=None, ax=None, label_names=None, linewidth=2, figsize=None, fontsize=10):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """

    # label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    if label_names is not None:
        label_names = list(label_names) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    if ax is None:
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img.astype(np.uint8))

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=linewidth))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])

        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0], ': '.join(caption), style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0}, fontsize=fontsize)

    return ax


# %% [markdown]
# # Creator Tool

# %%
def _get_inside_index(anchor, H, W): # Calc indicies of anchors which are located completely inside of the image whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) & # y0
        (anchor[:, 1] >= 0) & # x0
        (anchor[:, 2] <= H) & # y1
        (anchor[:, 3] <= W) # x1
    )[0]
    return index_inside

# if __name__=='__main__':
#     anchor = np.array([[-0.1,-0.1,0.9,0.9], [0,0,1,1], [0.1,0.1,1.1,1.1], [0.2,0.2,0.5,0.5]], dtype=np.float32)
#     H, W = 1, 1
#     index_inside = _get_inside_index(anchor, H, W)
#     print(f'index_inside = {index_inside}')
#     print(f'type of index_inside = {type(index_inside)}')

# %%
def _calc_ious(anchor, bbox): # ious between the anchors and the gt boxes
    ious = bbox_iou(anchor, bbox) # (A, 4) x (B, 4) -> (A, B)

    argmax_ious = ious.argmax(axis=1) # (A,): the indexes of bbox that is closest to each anchor
    max_ious = ious[np.arange(ious.shape[0]), argmax_ious] # (A,)

    gt_argmax_ious = ious.argmax(axis=0) # (B,)
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] # (B,)

    # gt_argmax_ious_all = np.where(ious == gt_max_ious)[0] # (B + alpha,)
    gt_argmax_ious_all = np.where((ious == gt_max_ious) & (gt_max_ious > 0))[0] # (B + alpha,) # BRAD BUG FIX: 2024-06-29

    return argmax_ious, max_ious, gt_argmax_ious, gt_max_ious, gt_argmax_ious_all

# if __name__=='__main__':
#     anchor = np.array([[0,0,1,1], [0.5,0.5,1.5,1.5], [1,1,2,2]], dtype=np.float32)
#     bbox = np.array([[0.9,0.9,1.9,1.9], [0.2,0.2,1.2,1.2]], dtype=np.float32)
#     print(f'anchor = {anchor}')
#     print(f'bbox = {bbox}')

#     argmax_ious, max_ious, gt_argmax_ious, gt_max_ious, gt_argmax_ious_all = _calc_ious(anchor, bbox)
#     print(f'argmax_ious = {argmax_ious}')
#     print(f'max_ious = {max_ious}')
#     print(f'gt_argmax_ious = {gt_argmax_ious}')
#     print(f'gt_max_ious = {gt_max_ious}')
#     print(f'gt_argmax_ious_all = {gt_argmax_ious_all}')
#     print(np.where(np.array([[True, False,True],[False, True,True],[True, True,False]])))

# %%
def _unmap(data, count, index, fill=0): # Unmap a subset of item (data) back to the original set of items (of size count)
    if len(data.shape) == 1: # e.g. label
        ret = np.empty((count,), dtype=data.dtype) # initialize
        ret.fill(fill)
        ret[index] = data
    else: # e.g. bboxes
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype) # initialize
        ret.fill(fill)
        ret[index, :] = data
    return ret

# if __name__=='__main__':
#     data = np.array([[1,1], [3,3], [5,5]])
#     count = 6
#     index = np.array([1,3,5])
#     ret = _unmap(data, count, index, fill=0)
#     print(f'ret = {ret}')

# %%
class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the sampled regions.

    """

    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5, use_original_subsample_for_postive_labels=True):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
        self.use_original_subsample_for_postive_labels = use_original_subsample_for_postive_labels

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.
        Types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which is a tuple of height and width of an image.

        Returns:
            (array, array):
            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape is :math:`(S,)`.
        """

        IMG_H, IMG_W = img_size

        if PATCH_REMOVE_INSIDE_INDEX_LIMIT:
            inside_index = np.arange(anchor.shape[0])
        else:
            inside_index = _get_inside_index(anchor, IMG_H, IMG_W)

        anchor_ = anchor[inside_index]
        label_, argmax_ious = self._create_label(anchor_, bbox)
        loc_ = bbox2loc(anchor_, bbox[argmax_ious]) # compute bounding box regression targets

        # map up to original set of anchors
        label = _unmap(label_, len(anchor), inside_index, fill=-1)
        loc = _unmap(loc_, len(anchor), inside_index, fill=0)

        return loc, label

    def _create_label(self, anchor, bbox): # label: 1 is positive, 0 is negative, -1 is dont care
        argmax_ious, max_ious, _, _, gt_argmax_ious_all = _calc_ious(anchor, bbox)

        label = np.empty((anchor.shape[0],), dtype=np.int32)
        label.fill(-1) # initialize with 'dont care'

        # ref) https://arxiv.org/pdf/1612.03144 (see 4.1. Feature Pyramid Networks for RPN in page 4)
        label[max_ious < self.neg_iou_thresh] = 0 # assign negative labels first so that positive labels can clobber them
        label[gt_argmax_ious_all] = 1 # positive label: for each gt, anchor with highest iou
        label[max_ious >= self.pos_iou_thresh] = 1 # positive label: above threshold IOU

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            if self.use_original_subsample_for_postive_labels:
                disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
                label[disable_index] = -1

            else: # Brad's algorithm
                i2i = np.empty((len(max_ious),), dtype=np.int32) # index-to-index vector
                i2i.fill(-1) # initialize
                i2i[pos_index] = range(len(pos_index))
                assert len(np.where(i2i[gt_argmax_ious_all] < 0)[0]) == 0 # make sure there is no '-1', which was used for initialization. Assume that gt_argmax_ious_all is a subset of pos_index
                # print(f'i2i = {i2i}')

                pos_ious = max_ious[pos_index].copy() # subgroup of max_ious
                pos_ious[i2i[gt_argmax_ious_all]] = 1.0 # forcefully assign the highest iou
                # print(f'pos_ious = {pos_ious}')

                disable_index = np.argsort(pos_ious)[:-n_pos] # pick the rest, outside top 'n_pos'
                label[pos_index[disable_index]] = -1 # assign 'dont care'
                # print(f'label = {label}')

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return label, argmax_ious

# if __name__=='__main__': # Brad
#     anchor_target_creator = AnchorTargetCreator()

#     anchor = np.array([[-1,-1,0,0], [0,0,1,1], [0.5,0.5,1.5,1.5], [1,1,2,2], [9,9,10,10]], dtype=np.float32)
#     bbox = np.array([[0.9,0.9,1.9,1.9], [0.2,0.2,1.2,1.2]], dtype=np.float32)
#     img_size = (10, 10)

#     loc, label = anchor_target_creator(bbox, anchor, img_size)
#     print(f'loc = {loc}')
#     print(f'label = {label}')


# %%
class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background if IoU is in [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample. Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes. Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        roi = torch.cat((roi, bbox), dim=0)

        iou = bbox_iou_torch(roi, bbox)
        gt_assignment = torch.argmax(iou, dim = 1) # (N,) of ground truth box index for each proposal
        max_iou = torch.max(iou, dim = 1).values # (N,) of maximum IoUs for each of the N proposals
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(round(self.n_sample * self.pos_ratio), pos_index.size(0)))
        if pos_index.size(0) > 0:
            pos_index = pos_index[torch.randperm(pos_index.size(0))[:pos_roi_per_this_image]]

        # Select background RoIs as those within [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = torch.where((self.neg_iou_thresh_lo <= max_iou) & (max_iou < self.neg_iou_thresh_hi))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size(0)))
        if neg_index.size(0) > 0:
            neg_index = neg_index[torch.randperm(neg_index.size(0))[:neg_roi_per_this_image]]

        # The indices that we're selecting (both positive and negative).
        keep_index = torch.cat([pos_index, neg_index])
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc_torch(sample_roi, bbox[gt_assignment[keep_index]])
        if torch.cuda.is_available(): # Brad
            _loc_normalize_mean_ = torch.tensor(loc_normalize_mean, dtype = torch.float32, device = "cuda")
            _loc_normalize_std_ = torch.tensor(loc_normalize_std, dtype = torch.float32, device = "cuda")
        else:
            _loc_normalize_mean_ = torch.tensor(loc_normalize_mean, dtype = torch.float32, device = "cpu")
            _loc_normalize_std_ = torch.tensor(loc_normalize_std, dtype = torch.float32, device = "cpu")
        gt_roi_loc = (gt_roi_loc - _loc_normalize_mean_) / _loc_normalize_std_

        return sample_roi, gt_roi_loc, gt_roi_label # (S, 4), (S, 4) encoded & normalized with mean, std, (S,) cf. S ~= n_sample

# if __name__=='__main__': # Brad
#     proposal_target_creator = ProposalTargetCreator()

#     roi = np.array([[0.1,0.1,1.1,1.1], [3,3,4,4]])
#     bbox = np.array([[0,0,1,1], [2,2,3,3], [4,4,5,5]])
#     label = np.array([1, 0, 2])

#     sample_roi, gt_roi_loc, gt_roi_label = proposal_target_creator(roi, bbox, label)
#     print(f'sample_roi = {sample_roi}')
#     print(f'gt_roi_loc = {gt_roi_loc}')
#     print(f'gt_roi_label = {gt_roi_label}')

# %%
class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets to a set of anchors.

    This class takes parameters to control number of bounding boxes to pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`, always use NMS in CPU mode. If :obj:`False`, the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on discarding bounding boxes based on their sizes.
    """

    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors. Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors. Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`, which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        roi = loc2bbox(anchor, loc) # Brad: roi = decoded, predicted bbox

        if PATCH_REMOVE_INSIDE_INDEX_LIMIT_2:
            pass
        else:
            # Clip predicted boxes to image.
            roi[:,0] = torch.clamp(roi[:,0], min = 0) # 0 <= y
            roi[:,1] = torch.clamp(roi[:,1], min = 0) # 0 <= x
            roi[:,2] = torch.clamp(roi[:,2], max = img_size[0]) # y <= img_size[0]
            roi[:,3] = torch.clamp(roi[:,3], max = img_size[1]) # x <= img_size[1]

        # Remove predicted boxes with either height or width < threshold.
        hs = roi[:,2] - roi[:,0] # y
        ws = roi[:,3] - roi[:,1] # x
        keep = torch.where((hs >= self.min_size) & (ws >= self.min_size))[0] # Brad: 2024-12-07
        roi = roi[keep]
        score = score[keep]

        # # Sort all (proposal, score) pairs by score from highest to lowest.
        # # Take top pre_nms_topN (e.g. 6000).
        # Keep only the top-N scores. Note that we do not care whether the
        # proposals were labeled as objects (score > 0.5) and peform a simple
        # ranking among all of them. Restricting them has a strong adverse impact
        # on training performance.
        order = torch.argsort(score)                   # sort in ascending order of objectness score
        order = order.flip(dims = (0,))               # descending order of score
        roi = roi[order][0:n_pre_nms]  # grab the top-N best proposals
        score = score[order][0:n_pre_nms]  # corresponding scores

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).
        keep = nms(
            boxes = roi,
            scores = score,
            iou_threshold = self.nms_thresh
            )
        keep = keep[0:n_post_nms]
        roi = roi[keep]

        return roi

# if __name__=='__main__': # Brad
#     parent_model = nn.Conv2d(3,3, 3,1,0)
#     print(f'parent_model.training = {parent_model.training}')

#     proposal_creator = ProposalCreator(parent_model, nms_thresh=0.7, min_size=0.5)
#     # proposal_creator = ProposalCreator(parent_model, nms_thresh=0.999, min_size=0.0)

#     loc = np.array([[0,0,0,0], [0.05,0.05,0,0], [0.5,0.5,0,0]], dtype=np.float32)
#     score = np.array([0.8, 0.9, 0.7], dtype=np.float32)
#     anchor = np.array([[0,0,1,1], [0,0,1,1], [0,0,1,1]], dtype=np.float32)
#     img_size = (2, 2)
    
#     roi = proposal_creator(loc, score, anchor, img_size)
#     print(f'roi = {roi}')

# %% [markdown]
# # Region Proposal Network

# %%
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # np = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1) # yxyx, (height*width, 4)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2)) # (1, A, 4) + (K, 1, 4) -> (K, A, 4)
    anchor = anchor.reshape((K * A, 4)).astype(np.float32) # (K, A, 4) -> (K*A, 4)
    return anchor # (K*A, 4)

# if __name__=='__main__': # Brad
#     # anchor_base = generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32])
#     anchor_base = generate_anchor_base(base_size=1, ratios=[1], anchor_scales=[1])
#     feat_stride, height, width = 16, 2, 3
#     anchor = _enumerate_shifted_anchor(anchor_base, feat_stride, height, width)
#     assert len(anchor) == len(anchor_base) * height * width
#     print(anchor)

# %%
def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

# if __name__=='__main__': # Brad
#     conv = nn.Conv2d(1, 1, 3, 1, 0)
#     print(f'weight = {conv.weight}')
#     print(f'bias = {conv.bias}')
#     normal_init(conv, 0, 0.01)
#     print(f'weight = {conv.weight}')
#     print(f'bias = {conv.bias}')    

# %%
class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference window.
        feat_stride (int): Stride size after extracting features from an image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512,
            base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None,
            feat_stride=16,

            # proposal_creator_params=dict(),
            # parameters of ProposalCreator()
            nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16,
    ):
        super().__init__()
        
        self.anchor_base = generate_anchor_base(base_size=base_size, ratios=ratios, anchor_scales=anchor_scales, y_offsets=y_offsets, x_offsets=x_offsets, num_offsets=num_offsets)
        self.feat_stride = feat_stride

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.loc = nn.Conv2d(mid_channels, self.anchor_base.shape[0] * 4, 1, 1, 0)
        self.score = nn.Conv2d(mid_channels, self.anchor_base.shape[0] * 2, 1, 1, 0)
        self.proposal_creator = ProposalCreator(self, nms_thresh=nms_thresh, n_train_pre_nms=n_train_pre_nms, n_train_post_nms=n_train_post_nms, n_test_pre_nms=n_test_pre_nms, n_test_post_nms=n_test_post_nms, min_size=min_size)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

    def forward(self, features, img_size):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            features (~torch.autograd.Variable): The Features extracted from images. Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`, which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of proposal boxes.  This is a concatenation of bounding box arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted bounding boxes from the :math:`i` th image, :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. Its shape is :math:`(H W A, 4)`.
        """
        N, _, H, W = features.shape # Brad: (N, C_in, H, W)
        # n_anchor = anchor.shape[0] // (H * W) # A = n_anchor
        A = self.anchor_base.shape[0]
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, H, W) # (H*W*A, 4)

        h = F.relu(self.conv1(features)) # Brad: (N, C_mid, H, W)

        #--- location ------------------------------------
        rpn_locs = self.loc(h) # Brad: (N, C_mid, H, W) -> (N, A*4, H, W)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(N, -1, 4) # Brad: (N, A*4, H, W) -> (N, H, W, A*4) -> (N, H*W*A, 4)

        #--- score ---------------------------------------
        rpn_scores = self.score(h) # Brad: (N, C_mid, H, W) -> (N, A*2, H, W)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() # Brad: (N, A*2, H, W) -> (N, H, W, A*2)
        rpn_softmax_scores = F.softmax(rpn_scores.view(N, H, W, A, 2), dim=4) # Brad: (N, H, W, A*2) -> (N, H, W, A, 2)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous() # Brad: (N, H, W, A, 2) -> (N, H, W, A)
        rpn_fg_scores = rpn_fg_scores.view(N, -1) # Brad: (N, H, W, A) -> (N, H*W*A)
        rpn_scores = rpn_scores.view(N, -1, 2) # Brad: (N, H, W, A*2) -> (N, H*W*A, 2)

        #--- proposal creator ----------------------------
        rois = list()
        roi_indices = list()
        for i in range(N):
            roi = self.proposal_creator(
                rpn_locs[i].detach(), # Brad: (N, H*W*A, 4) -> (H*W*A, 4)
                rpn_fg_scores[i].detach(), # Brad: (N, H*W*A) -> (H*W*A,)
                to_tensor(anchor), # (H*W*A, 4)
                img_size,
                )
            rois.append(roi)
            roi_indices.append(torch.full(size=(len(roi),), fill_value=i, dtype=torch.int32)) # UPGRADE_MULTI_BATCH
        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor # (N, H*W*A, 4), (N, H*W*A, 2), (K, 4), (K,), (H*W*A, 4) # K <= N*H*W*A

# if __name__=='__main__': # Brad
#     rpn = RegionProposalNetwork()

#     features = torch.tensor(np.ones((1,512,16,16), dtype=np.float32))
#     img_size = (256,256)
#     rpn_locs, rpn_scores, rois, roi_indices, anchor = rpn(features, img_size)
#     print(f'rpn_locs.shape = {rpn_locs.shape}')
#     print(f'rpn_scores.shape = {rpn_scores.shape}')
#     print(f'rois.shape = {rois.shape}')
#     print(f'roi_indices.shape = {roi_indices.shape}')
#     print(f'anchor.shape = {anchor.shape}') 

# %% [markdown]
# # Faster RCNN

# %%
class FeatureExtractor_VGG16(nn.Module): # PATCH_FEATURE_PYRAMID
  def __init__(self, model, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=False):
    super().__init__()

    #--- decompose: features ------------------------------------------------
    self.features = list(model.features)[:30] # Brad: only exclude the last layer (i.e. MaxPool) of the feature part

    #--- freeze parameters --------------------------------------------------
    for layer in self.features[:freeze_param_up_to]: # freeze top4 conv (= freeze_param_up_to=10)
        for p in layer.parameters():
            p.requires_grad = False

    #--- change ceil_mode of MaxPool2d --------------------------------------
    if ceil_mode_for_MaxPool2d:
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                layer.ceil_mode = ceil_mode_for_MaxPool2d
    return

  def forward(self, img):
    x = img
    for i in range(len(self.features)):
      x = self.features[i](x)
    out = x
    return {'out':out}

def decom_vgg16(use_pretrained_weights=True, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=False):
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        # model = vgg16(pretrained=False)
        model = vgg16(weights=None) # Brad update: 2024-10-12
        if use_pretrained_weights and (not opt.load_path):
            model.load_state_dict(torch.load(opt.caffe_pretrain_path))
    else:
        # model = vgg16(not opt.load_path)
        model = vgg16(weights='IMAGENET1K_V1' if use_pretrained_weights else None) # Brad update: 2024-10-12

    if PATCH_FEATURE_PYRAMID:
       features = FeatureExtractor_VGG16(model, freeze_param_up_to=freeze_param_up_to, ceil_mode_for_MaxPool2d=ceil_mode_for_MaxPool2d)
    else:
      #--- decompose: features ------------------------------------------------
      features = list(model.features)[:30] # Brad: only exclude the last layer (i.e. MaxPool) of the feature part

      in_channels, mid_channels = 512, 512

      #--- freeze parameters --------------------------------------------------
      for layer in features[:freeze_param_up_to]: # freeze top4 conv (= freeze_param_up_to=10)
          for p in layer.parameters():
              p.requires_grad = False

      #--- change ceil_mode of MaxPool2d --------------------------------------
      if ceil_mode_for_MaxPool2d:
          for layer in features:
              if isinstance(layer, nn.MaxPool2d):
                  layer.ceil_mode = ceil_mode_for_MaxPool2d

    #--- decompose: classifier ----------------------------------------------
    classifier = list(model.classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]

    classifier = [Flatten()] + classifier

    out_classifier = 4096

    if PATCH_FEATURE_PYRAMID:
      return features, nn.Sequential(*classifier), in_channels, mid_channels, out_classifier
    else:
      return nn.Sequential(*features), nn.Sequential(*classifier), in_channels, mid_channels, out_classifier

# if __name__=='__main__': # Brad
#     kwargs = {'env':'fasterrcnn', 'num_workers':0, 'test_num_workers':0, 'voc_data_dir':r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007'}
#     test = Config()
#     test._parse(kwargs, verbose=False)
#     print(test.load_path)

#     model = vgg16(weights='IMAGENET1K_V1')
#     print(f'model = {model}')

#     features, classifier, in_channels, mid_channels, out_classifier = decom_vgg16()
#     print(f'features = {features}')
#     print(f'classifier = {classifier}')

# %%
# model = vgg16(weights='IMAGENET1K_V1')
# # model = vgg19(weights='IMAGENET1K_V1')
# features = list(model.features)
# classifier = list(model.classifier)

# x = torch.zeros((1,3,224,224), dtype=torch.float32)
# for i, ss in enumerate(features):
#     x = ss(x)
#     print(f'[features {i}] x.shape = {x.shape}')

# x = model.avgpool(x)
# print(f'[avgpool] x.shape = {x.shape}')

# for i, ss in enumerate(classifier):
#     x = ss(x)
#     print(f'[classifier {i}] x.shape = {x.shape}')

# %%
class FeatureExtractor_RESNET(nn.Module): # Brad
  # def __init__(self, resnet, ceil_mode_for_MaxPool2d=False, freeze_param_up_to=None, always_freeze_first_batch_norm=False):
  def __init__(self, resnet, use_layers_up_to=7, ceil_mode_for_MaxPool2d=False, freeze_param_up_to=None, always_freeze_first_batch_norm=False): # PATCH_FEATURE_PYRAMID
    super().__init__()

    self.use_layers_up_to = use_layers_up_to
    self.freeze_param_up_to = freeze_param_up_to
    self.always_freeze_first_batch_norm = always_freeze_first_batch_norm

    if PATCH_FEATURE_PYRAMID:
      self.freeze_param_up_to = self.freeze_param_up_to if self.freeze_param_up_to is not None else 5 # i.e. default value = up to layer1

      #--- all layers in resnet -----------------------
      # ref) https://arxiv.org/pdf/1512.03385 (see Table 1 in page 5)
      self.conv1 = resnet.conv1 # conv1
      self.bn1 = resnet.bn1 # conv1
      self.relu = resnet.relu # conv1
      self.maxpool = resnet.maxpool # conv2_x
      self.layer1 = resnet.layer1 # conv2_x
      self.layer2 = resnet.layer2 # conv3_x
      self.layer3 = resnet.layer3 # conv4_x
      self.layer4 = resnet.layer4 # conv5_x
      # self.avgpool = resnet.avgpool
      # self.fc = resnet.fc

      #--- Freeze initial layers ----------------------
      if self.freeze_param_up_to >= 1:
        self._freeze(self.conv1)
      if self.freeze_param_up_to >= 2 or always_freeze_first_batch_norm: # !!!!!!
        self._freeze(self.bn1)
      if self.freeze_param_up_to >= 3:
        self._freeze(self.relu) # actually no effect, since there is no parameter
      if self.freeze_param_up_to >= 4:
        self._freeze(self.maxpool) # actually no effect, since there is no parameter
      if self.freeze_param_up_to >= 5:
        self._freeze(self.layer1)
      if self.freeze_param_up_to >= 6:
        self._freeze(self.layer2)
      if self.freeze_param_up_to >= 7:
        self._freeze(self.layer3)
      if self.freeze_param_up_to >= 8:
        self._freeze(self.layer4)

      #--- Ensure that all batchnorm layers are frozen, as described in Appendix A of [1]
      # self._freeze_batchnorm(self)
      for child in self.modules():
        if type(child) == nn.BatchNorm2d:
          self._freeze(layer = child)

      #--- change ceil_mode of MaxPool2d --------------------------------------
      if ceil_mode_for_MaxPool2d: # Brad: 2024-12-08
          for layer in self.modules(): # Brad: 2024-12-08
              if isinstance(layer, nn.MaxPool2d): # Brad: 2024-12-08
                  layer.ceil_mode = True # Brad: 2024-12-08

    else:
      # Feature extractor layers
      self._feature_extractor = nn.Sequential(
        resnet.conv1,     # 0
        resnet.bn1,       # 1
        resnet.relu,      # 2
        resnet.maxpool,   # 3
        resnet.layer1,    # 4
        resnet.layer2,    # 5
        resnet.layer3     # 6
      )

      # Freeze initial layers
      if freeze_param_up_to is None: # original code
        self._freeze(resnet.conv1)
        self._freeze(resnet.bn1)
        self._freeze(resnet.layer1)
      else:
        if freeze_param_up_to >= 1:
          self._freeze(resnet.conv1)
        if freeze_param_up_to >= 2 or always_freeze_first_batch_norm:
          self._freeze(resnet.bn1)
        if freeze_param_up_to >= 3:
          self._freeze(resnet.relu) # actually no effect, since there is no parameter
        if freeze_param_up_to >= 4:
          self._freeze(resnet.maxpool) # actually no effect, since there is no parameter
        if freeze_param_up_to >= 5:
          self._freeze(resnet.layer1)
        if freeze_param_up_to >= 6:
          self._freeze(resnet.layer2)
        if freeze_param_up_to >= 7:
          self._freeze(resnet.layer3)

      # Ensure that all batchnorm layers are frozen, as described in Appendix A of [1]
      self._freeze_batchnorm(self._feature_extractor)

      #--- change ceil_mode of MaxPool2d --------------------------------------
      if ceil_mode_for_MaxPool2d: # Brad: 2024-12-08
          for layer in self._feature_extractor: # Brad: 2024-12-08
              if isinstance(layer, nn.MaxPool2d): # Brad: 2024-12-08
                  layer.ceil_mode = True # Brad: 2024-12-08
    return

  # Override nn.Module.train()
  def train(self, mode = True):
    super().train(mode)

    #
    # During training, set all frozen blocks to evaluation mode and ensure that
    # all the batchnorm layers are also in evaluation mode. This is extremely
    # important and neglecting to do this will result in severely degraded
    # training performance.
    #
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()

      if PATCH_FEATURE_PYRAMID:
        if self.freeze_param_up_to >= 1:
          self._freeze(self.conv1.eval())
        if self.freeze_param_up_to >= 2 or self.always_freeze_first_batch_norm: # !!!!!!
          self._freeze(self.bn1.eval())
        if self.freeze_param_up_to >= 3:
          self._freeze(self.relu.eval()) # actually no effect, since there is no parameter
        if self.freeze_param_up_to >= 4:
          self._freeze(self.maxpool.eval()) # actually no effect, since there is no parameter
        if self.freeze_param_up_to >= 5:
          self._freeze(self.layer1.eval())
        if self.freeze_param_up_to >= 6:
          self._freeze(self.layer2.eval())
        if self.freeze_param_up_to >= 7:
          self._freeze(self.layer3.eval())
        if self.freeze_param_up_to >= 8:
          self._freeze(self.layer4.eval())

        # *All* batchnorm layers in eval mode
        self.apply(set_bn_eval)

      else:
        # Set fixed blocks to be in eval mode
        if self.freeze_param_up_to is None: # original code
          self._feature_extractor.eval()
          self._feature_extractor[5].train()
          self._feature_extractor[6].train()
        else:
          self._feature_extractor.train()
          if self.freeze_param_up_to >= 1:
            self._freeze(self._feature_extractor[0].eval())
          if self.freeze_param_up_to >= 2 or self.always_freeze_first_batch_norm:
            self._freeze(self._feature_extractor[1].eval())
          if self.freeze_param_up_to >= 3:
            self._freeze(self._feature_extractor[2].eval())
          if self.freeze_param_up_to >= 4:
            self._freeze(self._feature_extractor[3].eval())
          if self.freeze_param_up_to >= 5:
            self._freeze(self._feature_extractor[4].eval())
          if self.freeze_param_up_to >= 6:
            self._freeze(self._feature_extractor[5].eval())
          if self.freeze_param_up_to >= 7:
            self._freeze(self._feature_extractor[6].eval())

        # *All* batchnorm layers in eval mode
        self._feature_extractor.apply(set_bn_eval)
    return

  def forward(self, img):
    if PATCH_FEATURE_PYRAMID:
      c1 = self.conv1(img)  if self.use_layers_up_to >= 1 else None
      c1 = self.bn1(c1)     if self.use_layers_up_to >= 2 else None
      c1 = self.relu(c1)    if self.use_layers_up_to >= 3 else None

      c2 = self.maxpool(c1) if self.use_layers_up_to >= 4 else None
      c2 = self.layer1(c2)  if self.use_layers_up_to >= 5 else None

      c3 = self.layer2(c2)  if self.use_layers_up_to >= 6 else None

      c4 = self.layer3(c3)  if self.use_layers_up_to >= 7 else None

      c5 = self.layer4(c4)  if self.use_layers_up_to >= 8 else None

      out = [c for c in [c1,c2,c3,c4,c5] if c is not None][-1] # the last layer output, which is not None

      return {'out':out, 'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4, 'c5':c5}

    else:
      y = self._feature_extractor(img)
      return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


class PoolToFeatureVector_RESNET(nn.Module): # Brad
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

    self._avgpool = resnet.avgpool # Brad: 2024-12-10
    self._flatten = Flatten() # Brad: 2024-12-10

  def train(self, mode = True):
    # See comments in FeatureVector.train()
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

    # Average together the last two dimensions to remove them -> (N, 2048).
    # It is also possible to max pool, e.g.:
    # y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
    # This may even be better (74.96% mAP for ResNet50 vs. 73.2% using the
    # current method).
    # y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048) # Brad: 2024-12-10
    y = self._avgpool(y) # Brad: 2024-12-10
    y = self._flatten(y) # Brad: 2024-12-10
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


def set_stride_1x1_of_all_children(node, path=[]): # recursively find child nodes and alter stride, if necessary
    for name, child in node.named_children():
        path.append(name)
        if 'stride' in child.__dict__ and child.stride != (1,1):
            old_stride = child.stride
            child.stride = (1,1)
            print(f'... alter stride? [{" >> ".join(path)}] changed stride from {old_stride} to {child.stride}!!!')
        set_stride_1x1_of_all_children(child, path)
        _ = path.pop()
    return

def decom_resnet(
    backbone_name='resnet50', use_pretrained_weights=True, freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True, always_freeze_first_batch_norm=False,
    # set_stride_1x1=True,
    set_stride_1x1=False,
    _original_sheme_=False,
    ):
    #--- load model (with pretrained weights) --------------------------------
    if backbone_name=='resnet50':
        model = resnet50(weights="DEFAULT" if use_pretrained_weights else None)
    elif backbone_name=='resnet101':
        model = resnet101(weights="DEFAULT" if use_pretrained_weights else None)
    elif backbone_name=='resnet152':
        model = resnet152(weights="DEFAULT" if use_pretrained_weights else None)

    elif backbone_name=='resnext50_32x4d':
        model = resnext50_32x4d(weights="DEFAULT" if use_pretrained_weights else None)
    elif backbone_name=='resnext101_32x8d':
        model = resnext101_32x8d(weights="DEFAULT" if use_pretrained_weights else None)
    elif backbone_name=='resnext101_64x4d':
        model = resnext101_64x4d(weights="DEFAULT" if use_pretrained_weights else None)

    else:
        raise(Exception(f'Brad error: no such model: {backbone_name}...'))

    #--- decompose: features ------------------------------------------------
    # features, in_channels = list(model.children())[:-3], 1024
    if _original_sheme_:
      features = OrderedDict((n,c) for n,c in model.named_children() if n not in ['layer4','avgpool','fc']) # up to 4th stage, i.e. layer3
    else:
      features = FeatureExtractor_RESNET(resnet = model, ceil_mode_for_MaxPool2d=ceil_mode_for_MaxPool2d, freeze_param_up_to=freeze_param_up_to, always_freeze_first_batch_norm=always_freeze_first_batch_norm) # Brad: 2024-12-10

    in_channels, mid_channels = 1024, 1024

    #--- decompose: classifier ----------------------------------------------
    if _original_sheme_:
      classifier = [model.layer4, model.avgpool, Flatten()] # Brad: 2024-12-10
    else:
      classifier = PoolToFeatureVector_RESNET(resnet = model) # Brad: 2024-12-10
    
    if set_stride_1x1:
        set_stride_1x1_of_all_children(classifier[0][0]) # alter the stride of the first layer, i.e. [0], of the 5th stage, i.e. 'layer4'

    out_classifier = 2048

    #--- freeze parameters --------------------------------------------------
    # for layer in features[:freeze_param_up_to]: # freeze top4 conv (= freeze_param_up_to=10)
    #     for p in layer.parameters():
    #         p.requires_grad = False
    if _original_sheme_: # Brad: 2024-12-10
        freeze, found = True, False
        for name, layer in features.items():
            for n,p in layer.named_parameters():
                full_name = name + '.' + n

                if (freeze_param_up_to is not None) and found and not (len(freeze_param_up_to) <= len(full_name) and freeze_param_up_to == full_name[:len(freeze_param_up_to)]):
                    freeze = False # turn off freezing

                if (freeze_param_up_to is not None) and freeze:
                    p.requires_grad = False # freeze parameters

                print(f'... trainable? {full_name}.requires_grad = {p.requires_grad}{"" if p.requires_grad else " [FREEZED]"}')
                if (freeze_param_up_to is not None) and (len(freeze_param_up_to) <= len(full_name) and freeze_param_up_to == full_name[:len(freeze_param_up_to)]):
                    found = True
        if (freeze_param_up_to is not None) and not found:
            raise(Exception(f'Brad error: no paramter found... please check: "{freeze_param_up_to}"'))

    #--- change ceil_mode of MaxPool2d --------------------------------------
    if _original_sheme_: # Brad: 2024-12-10
      if ceil_mode_for_MaxPool2d:
          # for layer in features:
          for _, layer in features.items():
              if isinstance(layer, nn.MaxPool2d):
                  layer.ceil_mode = ceil_mode_for_MaxPool2d

    # return nn.Sequential(*features), nn.Sequential(*classifier), in_channels
    if _original_sheme_:
      return nn.Sequential(features), nn.Sequential(*classifier), in_channels, mid_channels, out_classifier # Brad: 2024-12-10
    else:
      return features, classifier, in_channels, mid_channels, out_classifier # Brad: 2024-12-10

# if __name__=='__main__':
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet50', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True)
#     features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet101', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet101', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True, set_stride_1x1=False)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet152', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnext50_32x4d', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnext101_32x8d', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnext101_64x4d', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet50', freeze_param_up_to='layer2', ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet50', freeze_param_up_to='layer2', ceil_mode_for_MaxPool2d=True)
#     # features, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name='resnet50', freeze_param_up_to=None, ceil_mode_for_MaxPool2d=True, set_stride_1x1=False)

#     x = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
#     print(f'x.shape = {x.shape}')
#     x = features(x)
#     print(f'x.shape = {x.shape}')
#     # assert x.shape[1] == in_channels

#     y = torch.zeros((1, 1024, 7, 7), dtype=torch.float32)
#     # y = torch.zeros((1, 1024, 14, 14), dtype=torch.float32)
#     print(f'y.shape = {y.shape}')
#     for i in range(len(classifier)):
#         y = classifier[i](y)
#         print(f'y.shape = {y.shape}')
#     assert y.shape[1] == out_classifier
#     print(f'in_channels={in_channels}, mid_channels={mid_channels}, out_classifier={out_classifier}')

# %%
class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16
    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier, out_classifier=4096, is_roi_align=False, roi_align_sampling_ratio=-1, roi_align_aligned=False):
        # n_class includes the background
        super().__init__()

        self.n_class = n_class

        if is_roi_align:
            self.roi = RoIAlign((roi_size, roi_size), spatial_scale, roi_align_sampling_ratio, roi_align_aligned) # https://pytorch.org/vision/main/generated/torchvision.ops.RoIAlign.html
        else:
            self.roi = RoIPool((roi_size, roi_size), spatial_scale) # https://pytorch.org/vision/main/generated/torchvision.ops.RoIPool.html
        self.classifier = classifier
        self.cls_loc = nn.Linear(out_classifier, n_class * 4)
        self.score = nn.Linear(out_classifier, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, features, rois, roi_indices): # (N, C, H, W), (K, 4), (K,) # K <= N*H*W*A
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            features (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """

        roi_indices, rois = to_tensor(roi_indices).float(), to_tensor(rois).float() # in case roi_indices is  ndarray
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1) # [:, None]: add one more dimension into dim=1: i.e. (K, 1+4) = (K, 5)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]] # NOTE: important: yx->xy
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(features, indices_and_rois) # (K, C, roi_size, roi_size) : https://pytorch.org/vision/main/generated/torchvision.ops.roi_pool.html#torchvision.ops.roi_pool

        fc7 = self.classifier(pool) # (K, C, roi_size, roi_size) -> Flatten:(K, C*roi_size*roi_size) -> (K, out_classifier)
        roi_cls_locs = self.cls_loc(fc7) # (K, n_class*4)
        roi_scores = self.score(fc7) # (K, n_class)

        return roi_cls_locs, roi_scores # (K, n_class*4), (K, n_class)

# if __name__=='__main__':
#     a = torch.tensor(np.array([[1,2,3],[4,5,6]]))
#     print(a.shape)
#     print(a[:,None].shape)

# %%
def find_first_last_indices_of_uniform_part(indexes, verbos=False):
    ijs, i = [], 0
    for j in range(len(indexes)):
        if (j+1 == len(indexes)) or (indexes[j] != indexes[j+1]): # if last element or if the current element is different from the next element
            ijs.append([i, j+1])
            i = j + 1 # update the initial index

            if verbos:
                i_, j_ = ijs[-1]
                print(f'i = {i_}, j = {j_}, indexes = {indexes[i_:j_]}')
    return ijs

# if __name__=='__main__':
#     indexes = torch.tensor([]).tolist()
#     indexes = torch.tensor([0,0,0,0]).tolist()
#     indexes = torch.tensor([0,0,0,1,1,1,1,2,2,3,3,3,4,5,5,5]).tolist()
#     ijs = find_first_last_indices_of_uniform_part(indexes, verbos=True)

# %%
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    if torch.cuda.is_available():
        in_weight = torch.zeros(gt_loc.shape).cuda()
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    else:
        in_weight = torch.zeros(gt_loc.shape) # Brad
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1 # Brad
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

# %%
class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference window.
    
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation of localization estimates.

    --- parameters for training model -----------------------------------------------------
    wrapper for conveniently training. return losses

        The losses include:

        * :obj:`rpn_loc_loss`: The localization loss for Region Proposal Network (RPN).
        * :obj:`rpn_cls_loss`: The classification loss for RPN.
        * :obj:`roi_loc_loss`: The localization loss for the head module.
        * :obj:`roi_cls_loss`: The classification loss for the head module.
        * :obj:`total_loss`: The sum of 4 loss above.

        Args:
            faster_rcnn (model.FasterRCNN):
                A Faster R-CNN model that is going to be trained.        
    """

    def __init__(
        self,
        n_fg_class=20, 
        feat_stride = 16,  # downsample 16x for output of conv5 in vgg16
        loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
        backbone_name='vgg16', use_pretrained_weights=True, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=False, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
        #--- parameters for region proposal module ---------------------------------
        base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None,
        nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16,   
        roi_size=7, is_roi_align=False, roi_align_sampling_ratio=-1, roi_align_aligned=False,
        #--- parameters for training model -----------------------------------------
        atc_n_sample=256, atc_pos_iou_thresh=0.7, atc_neg_iou_thresh=0.3, atc_pos_ratio=0.5, atc_use_original_subsample_for_postive_labels=True,
        ptc_n_sample=128, ptc_pos_ratio=0.25, ptc_pos_iou_thresh=0.5, ptc_neg_iou_thresh_hi=0.5, ptc_neg_iou_thresh_lo=0.0,
        ):

        super().__init__()
        self._init_inputs = {k:v for k,v in locals().items() if k not in ['self','__class__']} # save input arguments

        if backbone_name=='vgg16':
            extractor, classifier, in_channels, mid_channels, out_classifier = decom_vgg16(use_pretrained_weights=use_pretrained_weights, freeze_param_up_to=freeze_param_up_to, ceil_mode_for_MaxPool2d=ceil_mode_for_MaxPool2d)
        elif backbone_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d']:
            extractor, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name=backbone_name, use_pretrained_weights=use_pretrained_weights, freeze_param_up_to=freeze_param_up_to, ceil_mode_for_MaxPool2d=ceil_mode_for_MaxPool2d, always_freeze_first_batch_norm=always_freeze_first_batch_norm)
        else:
            raise(Exception(f'Brad error: not available backbone name: {backbone_name}...'))

        if use_untrained_vgg_classifier:
            classifier = [
                # nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.Linear(in_features=in_channels * (roi_size * roi_size), out_features=4096, bias=True), # Brad update: 2024-10-12
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
            ]
            classifier = [nn.Flatten()] + classifier

            classifier = nn.Sequential(*classifier)
            out_classifier = 4096
            print('Brad info: new classifier is built, instead of VGG16 classifier')

        self.extractor = extractor
        self.rpn = RegionProposalNetwork(in_channels, mid_channels, ratios=ratios, anchor_scales=anchor_scales, y_offsets=y_offsets, x_offsets=x_offsets, num_offsets=num_offsets, feat_stride=feat_stride, base_size=base_size, nms_thresh=nms_thresh, n_train_pre_nms=n_train_pre_nms, n_train_post_nms=n_train_post_nms, n_test_pre_nms=n_test_pre_nms, n_test_post_nms=n_test_post_nms, min_size=min_size)
        self.head = VGG16RoIHead(n_class=n_fg_class + 1, roi_size=roi_size, spatial_scale=(1. / feat_stride), classifier=classifier, out_classifier=out_classifier, is_roi_align=is_roi_align, roi_align_sampling_ratio=roi_align_sampling_ratio, roi_align_aligned=roi_align_aligned)

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

        #--- parameters for training model -----------------------------------------
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator(n_sample=atc_n_sample, pos_iou_thresh=atc_pos_iou_thresh, neg_iou_thresh=atc_neg_iou_thresh, pos_ratio=atc_pos_ratio, use_original_subsample_for_postive_labels=atc_use_original_subsample_for_postive_labels)
        self.proposal_target_creator = ProposalTargetCreator(n_sample=ptc_n_sample, pos_ratio=ptc_pos_ratio, pos_iou_thresh=ptc_pos_iou_thresh, neg_iou_thresh_hi=ptc_neg_iou_thresh_hi, neg_iou_thresh_lo=ptc_neg_iou_thresh_lo)

        self.optimizer = self.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(self.n_class)
        self._losses = {k:[] for k in ['total_loss','rpn_loc_loss','rpn_cls_loss','roi_loc_loss','roi_cls_loss']}

        # https://github.com/pytorch/vision/issues/223
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/2
        # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # return appr -1~1 RGB
        self.pytorch_normalze = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def pre_process(self, imgs):
        assert imgs.shape[-1] == 3 # (N, H, W, C): Channel = 3
        imgs = imgs.permute(0, 3, 1, 2) # torch.tensor: (N, H, W, C) -> (N, C, H, W)
        imgs = imgs.float()
        imgs = imgs / 255.
        imgs = self.pytorch_normalze(imgs)
        return imgs

    def forward(self, imgs):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            imgs (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is :math:`(R',)`.
        """
        imgs = self.pre_process(imgs) # (N, H, W, C) -> (N, C, H, W)

        img_size = imgs.shape[2:] # _, _, H, W = imgs.shape

        if PATCH_FEATURE_PYRAMID:
            out = self.extractor(imgs) # (N, C, H, W)
            features = out['out']
        else:
            features = self.extractor(imgs) # (N, C, H, W)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size) # (N, H*W*A, 4), (N, H*W*A, 2), (K, 4), (K,), (H*W*A, 4) # K <= N*H*W*A
        roi_cls_locs, roi_scores = self.head(features, rois, roi_indices) # (K, n_class*4), (K, n_class)
        return roi_cls_locs, roi_scores, rois, roi_indices # (K, n_class*4), (K, n_class), (K, 4), (K,)

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob): # only for 1-batch
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]

            mask = prob_l > self.score_thresh

            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]

            keep = nms(cls_bbox_l, prob_l, self.nms_thresh) # Brad: https://pytorch.org/vision/main/generated/torchvision.ops.nms.html

            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @torch.no_grad
    def predict(self, imgs, visualize=False, visualize_score_thresh=None):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            if visualize_score_thresh is not None:
                self.score_thresh = visualize_score_thresh

        _, IMG_H, IMG_W, _ = imgs.shape

        bboxes = list()
        labels = list()
        scores = list()
        roi_cls_loc_BATCH, roi_scores_BATCH, rois_BATCH, roi_indices_BATCH = self(imgs)
        indexes = roi_indices_BATCH.tolist()
        ijs = find_first_last_indices_of_uniform_part(indexes)
        for i, j in ijs:
            roi_cls_loc = roi_cls_loc_BATCH[i:j].detach()
            roi_scores = roi_scores_BATCH[i:j].detach()
            roi = to_tensor(rois_BATCH[i:j])

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            if torch.cuda.is_available():
                mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
                std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]
            else:
                mean = torch.Tensor(self.loc_normalize_mean).repeat(self.n_class)[None] # Brad
                std = torch.Tensor(self.loc_normalize_std).repeat(self.n_class)[None] # Brad

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4) # (S, C*4) -> (S, C, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc) # (S, C*4) -> (S, C, 4)
            cls_bbox = loc2bbox(roi.reshape(-1, 4), roi_cls_loc.reshape(-1, 4))
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            if PATCH_REMOVE_INSIDE_INDEX_LIMIT_3:
                pass
            else:
                # clip bounding box
                cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=IMG_H) # y
                cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=IMG_W) # x

            prob = (F.softmax(to_tensor(roi_scores), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob) # only for 1-batch
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    #--- for training model -----------------------------------------------------------
    def forward_to_train_model(self, imgs, bboxes, labels):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes. Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels. Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value is :math:`[0, L - 1]`. :math:`L` is the number of foreground classes.
            scale (float): Amount of scaling applied to the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        imgs = self.pre_process(imgs) # (N, H, W, C) -> (N, C, H, W)

        img_size = imgs.shape[2:] # _, _, H, W = imgs.shape

        if PATCH_FEATURE_PYRAMID:
            out = self.extractor(imgs) # (N, C, H, W)
            features = out['out']
        else:
            features = self.extractor(imgs) # (N, C, H, W)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size) # (N, H*W*A, 4), (N, H*W*A, 2), (K, 4), (K,), (H*W*A, 4) # K <= N*H*W*A

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        indexes = roi_indices.tolist()
        ijs = find_first_last_indices_of_uniform_part(indexes)
        sample_roi, sample_roi_index, gt_roi_loc, gt_roi_label = [], [], [], []
        for n, (bbox, label) in enumerate(zip(bboxes, labels)):
            i, j = ijs[n]
            roi = rois[i:j]
            sample_roi_, gt_roi_loc_, gt_roi_label_ = self.proposal_target_creator(roi.detach(), bbox.detach(), label.detach(), self.loc_normalize_mean, self.loc_normalize_std) # (S, 4), (S, 4) encoded & normalized with mean, std, (S,) cf. S ~= n_sample
            sample_roi.append(sample_roi_)
            sample_roi_index.append(torch.full(size=(len(sample_roi_),), fill_value=n, dtype=torch.int32))
            gt_roi_loc.append(gt_roi_loc_)
            gt_roi_label.append(gt_roi_label_)
        sample_roi = torch.concatenate(sample_roi, dim=0).detach()
        sample_roi_index = torch.concatenate(sample_roi_index, dim=0).detach()
        gt_roi_loc = torch.concatenate(gt_roi_loc, dim=0).detach()
        gt_roi_label = torch.concatenate(gt_roi_label, dim=0).detach()

        gt_roi_loc, gt_roi_label = to_tensor(gt_roi_loc), to_tensor(gt_roi_label).long() # update 20240707

        roi_cls_locs, roi_scores = self.head(features, sample_roi, sample_roi_index) # (K, n_class*4), (K, n_class) = actually, (S, n_class*4), (S, n_class)

        # ------------------ RPN losses -------------------#
        rpn_loc = rpn_locs.view(-1, 4) # (N, H*W*A, 4) -> (N*H*W*A, 4)
        rpn_score = rpn_scores.view(-1, 2) # (N, H*W*A, 2) -> (N*H*W*A, 2)

        gt_rpn_loc, gt_rpn_label = [], []
        for n, bbox in enumerate(bboxes):
            gt_rpn_loc_, gt_rpn_label_ = self.anchor_target_creator(to_numpy(bbox), anchor, img_size) # (H*W*A, 4), (H*W*A,)
            gt_rpn_loc.append(gt_rpn_loc_)
            gt_rpn_label.append(gt_rpn_label_)
        gt_rpn_loc = np.concatenate(gt_rpn_loc, axis=0)
        gt_rpn_label = np.concatenate(gt_rpn_label, axis=0)

        gt_rpn_loc, gt_rpn_label = to_tensor(gt_rpn_loc), to_tensor(gt_rpn_label).long()

        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma) # (H*W*A, 4) <-> (H*W*A, 4)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1) # NOTE: default value of ignore_index is -100 ...: (H*W*A, 2) <-> (H*W*A,)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = to_numpy(rpn_score)[to_numpy(gt_rpn_label) > -1]
        self.rpn_cm.add(to_tensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_locs.shape[0]

        roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4) # Brad: (K, n_class*4) -> (K, Class, 4)
        roi_loc = roi_cls_locs[torch.arange(0, n_sample).long(), gt_roi_label] # update 20240707, Brad: (K, Class, 4) -> (K, 4)

        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_scores, gt_roi_label) # Brad: (K, n_class) <=> (K,)

        self.roi_cm.add(to_tensor(roi_scores, False), gt_roi_label.data.long())

        return {
            'total_loss':rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss,
            'rpn_loc_loss':rpn_loc_loss,
            'rpn_cls_loss':rpn_cls_loss,
            'roi_loc_loss':roi_loc_loss,
            'roi_cls_loss':roi_cls_loss,
        }

    def train_step(self, imgs, bboxes, labels, max_norm_of_clip_grad_norm=None):
        self.optimizer.zero_grad()
        out = self.forward_to_train_model(imgs, bboxes, labels)
        out['total_loss'].backward()
        if max_norm_of_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm_of_clip_grad_norm)
        self.optimizer.step()
        self.update_meters(out)
        return out

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.state_dict()
        save_dict['model_init_inputs'] = self._init_inputs # Brad: 2024-12-30
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        # save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        if 'model' in state_dict:
            # self.__init__(**state_dict['model_init_inputs']) # Brad: 2024-12-30
            self.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.load_state_dict(state_dict)
            return None
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return None

    def update_meters(self, out):
        for k,v in out.items(): # Brad: 2024-12-14
            if k in self._losses: # only in the list of pre-defined losses
                self._losses.setdefault(k, []).append(v.item()) # NOTE: '.item()' needs to be included, I think

    def reset_meters(self):
        self._losses = {k:[] for k in self._losses.keys()} # reset losses, Brad: 2024-12-14
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k:(statistics.mean(vv) if vv else float('nan')) for k, vv in self._losses.items()} # Brad: 2024-12-14


# %% [markdown]
# # YOLOv2

# %%
TURN_ON_loss_bbox_as_is = False

LAMBDA_COORD, LAMBDA_NOOBJ = 1, 1
# LAMBDA_COORD, LAMBDA_NOOBJ = 0.001, 1
# LAMBDA_COORD, LAMBDA_NOOBJ = 5, 0.5

TURN_ON_include_confidence = True
TURN_ON_include_confidence_split_conf = True
TURN_ON_include_confidence_SIMPLE = False

TURN_ON_anchor_box_encoding = True
TURN_ON_include_outer_boxes = True
TURN_ON_gradient_clip_by_norm = 10
TURN_ON_anchor_selection_for_yolo = True
TURN_ON_loss_for_yolo = True

# %%
def bbox2loc_BETA(cells, src_bbox, dst_bbox):
    def inverse_sigmoid(y): # https://stackoverflow.com/questions/10097891/inverse-logistic-sigmoid-function
        return np.log(y) - np.log(1 - y)

    eps = np.finfo(np.float32).eps

    cy0 = cells[:, 0]
    cx0 = cells[:, 1]
    ch = cells[:, 2] - cells[:, 0]
    cw = cells[:, 3] - cells[:, 1]

    by = 0.5*(dst_bbox[:, 0] + dst_bbox[:, 2]) # center point of bbox
    bx = 0.5*(dst_bbox[:, 1] + dst_bbox[:, 3]) # center point of bbox
    bh = dst_bbox[:, 2] - dst_bbox[:, 0]
    bw = dst_bbox[:, 3] - dst_bbox[:, 1]

    ph = src_bbox[:, 2] - src_bbox[:, 0]
    pw = src_bbox[:, 3] - src_bbox[:, 1]

    ty = inverse_sigmoid(((by - cy0) / ch.clip(min=eps)).clip(min=eps, max=1-eps))
    tx = inverse_sigmoid(((bx - cx0) / cw.clip(min=eps)).clip(min=eps, max=1-eps))
    th = np.log(bh / ph.clip(min=eps))
    tw = np.log(bw / pw.clip(min=eps))

    loc = np.vstack((ty, tx, th, tw)).transpose()
    return loc


# %%
def loc2bbox_BETA(src_bbox, loc, cell_size_x=None, cell_size_y=None): # Brad: src_bbox = "reference bbox"
    if src_bbox.shape[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    cy = 0.5*(src_bbox[:,0] + src_bbox[:,2]) # (A,), center point
    cx = 0.5*(src_bbox[:,1] + src_bbox[:,3]) # (A,), center point

    cy0 = cy - 0.5*cell_size_y # y0
    cx0 = cx - 0.5*cell_size_x # x0
    if torch.cuda.is_available(): # Brad
        # ch = torch.full(cy0.shape, cell_size_y, dtype = cy0.dtype, device = "cuda")
        # cw = torch.full(cx0.shape, cell_size_x, dtype = cx0.dtype, device = "cuda")
        ch = torch.full(cy0.shape, cell_size_y, dtype = loc.dtype, device = "cuda")
        cw = torch.full(cx0.shape, cell_size_x, dtype = loc.dtype, device = "cuda")
    else:
        # ch = torch.full(cy0.shape, cell_size_y, dtype = cy0.dtype, device = "cpu")
        # cw = torch.full(cx0.shape, cell_size_x, dtype = cx0.dtype, device = "cpu")
        ch = torch.full(cy0.shape, cell_size_y, dtype = loc.dtype, device = "cpu")
        cw = torch.full(cx0.shape, cell_size_x, dtype = loc.dtype, device = "cpu")

    ph = src_bbox[:, 2] - src_bbox[:, 0] # Brad: y_max - y_min
    pw = src_bbox[:, 3] - src_bbox[:, 1] # Brad: x_max - x_min

    ty = loc[:, 0]
    tx = loc[:, 1]
    th = loc[:, 2]
    tw = loc[:, 3]

    by = torch.sigmoid(ty) * ch + cy0
    bx = torch.sigmoid(tx) * cw + cx0
    bh = torch.exp(th) * ph
    bw = torch.exp(tw) * pw

    if torch.cuda.is_available(): # Brad
        dst_bbox = torch.empty(loc.shape, dtype = loc.dtype, device = "cuda")
    else: # Brad
        dst_bbox = torch.empty(loc.shape, dtype = loc.dtype, device = "cpu") # Brad
    dst_bbox[:, 0] = by - 0.5 * bh # y_min
    dst_bbox[:, 1] = bx - 0.5 * bw # x_min
    dst_bbox[:, 2] = by + 0.5 * bh # y_max
    dst_bbox[:, 3] = bx + 0.5 * bw # x_max

    return dst_bbox

# if __name__=='__main__':
#     src_bbox = torch.tensor([[0,0,1,1]])
#     loc = torch.tensor([[0,0,0,0]])
    
#     dst_bbox = loc2bbox_BETA(src_bbox, loc, cell_size_x=1, cell_size_y=1)
#     print(f'dst_bbox = {dst_bbox}')

# %%
def _loss_for_YOLO(x, t, in_weight):
    diff = in_weight * (x - t)
    y = diff ** 2
    return y.sum()

def _fast_rcnn_loc_loss_BETA(pred_loc, gt_loc, gt_label, sigma):
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    if torch.cuda.is_available():
        in_weight = torch.zeros(gt_loc.shape).cuda()
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    else:
        in_weight = torch.zeros(gt_loc.shape) # Brad
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1 # Brad
    if TURN_ON_loss_for_yolo:
        loc_loss = _loss_for_YOLO(pred_loc, gt_loc, in_weight.detach())
    else:
        loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

# %%
def _fast_rcnn_loc_loss_BETA2(pred_loc, gt_bbox, gt_label, anchors, cell_size_x, cell_size_y):
    # pred_bbox = loc2bbox_BETA(anchors, pred_loc, cell_size_x=cell_size_x, cell_size_y=cell_size_y)

    cy = 0.5*(anchors[:,0] + anchors[:,2]) # (A,), center point
    cx = 0.5*(anchors[:,1] + anchors[:,3]) # (A,), center point

    cy0 = cy - 0.5*cell_size_y # y0
    cx0 = cx - 0.5*cell_size_x # x0
    ch = torch.full(cy0.shape, cell_size_y, dtype = pred_loc.dtype, device = ("cuda" if torch.cuda.is_available() else "cpu"))
    cw = torch.full(cx0.shape, cell_size_x, dtype = pred_loc.dtype, device = ("cuda" if torch.cuda.is_available() else "cpu"))

    ph = anchors[:, 2] - anchors[:, 0] # Brad: y_max - y_min
    pw = anchors[:, 3] - anchors[:, 1] # Brad: x_max - x_min

    ty = pred_loc[:, 0]
    tx = pred_loc[:, 1]
    th = pred_loc[:, 2]
    tw = pred_loc[:, 3]

    by = torch.sigmoid(ty) * ch + cy0
    bx = torch.sigmoid(tx) * cw + cx0
    bh = torch.exp(th) * ph
    bw = torch.exp(tw) * pw

    gy = 0.5*(gt_bbox[:,0] + gt_bbox[:,2])
    gx = 0.5*(gt_bbox[:,1] + gt_bbox[:,3])
    gh = gt_bbox[:,2] - gt_bbox[:,0]
    gw = gt_bbox[:,3] - gt_bbox[:,1]

    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    if torch.cuda.is_available():
        in_weight = torch.zeros(gt_bbox.shape).cuda()
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    else:
        in_weight = torch.zeros(gt_bbox.shape) # Brad
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1 # Brad

    in_weight = in_weight.detach()

    diff_y = in_weight[:, 0] * (by - gy) # y, x. https://arxiv.org/pdf/2304.00501
    diff_x = in_weight[:, 1] * (bx - gx) # y, x. https://arxiv.org/pdf/2304.00501
    diff_h = in_weight[:, 2] * (torch.sqrt(torch.abs(bh)) - torch.sqrt(torch.abs(gh))) # h, w. https://arxiv.org/pdf/2304.00501
    diff_w = in_weight[:, 3] * (torch.sqrt(torch.abs(bw)) - torch.sqrt(torch.abs(gw))) # h, w. https://arxiv.org/pdf/2304.00501
    y = diff_y ** 2 + diff_x ** 2 + diff_h ** 2 + diff_w ** 2

    # Normalize by total number of negtive and positive rois.
    loc_loss = y.sum() / ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

# %%
class RegionProposalNetwork_BETA(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference window.
        feat_stride (int): Stride size after extracting features from an image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512,
            base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None,
            feat_stride=16,

            num_score=2, # BRAD: 2025-01-20

            # proposal_creator_params=dict(),
            # parameters of ProposalCreator()
            nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16,
    ):
        super().__init__()
        
        self.anchor_base = generate_anchor_base(base_size=base_size, ratios=ratios, anchor_scales=anchor_scales, y_offsets=y_offsets, x_offsets=x_offsets, num_offsets=num_offsets)
        self.feat_stride = feat_stride

        self.num_score = num_score # BRAD: 2025-01-20

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.loc = nn.Conv2d(mid_channels, self.anchor_base.shape[0] * 4, 1, 1, 0)
        # self.score = nn.Conv2d(mid_channels, self.anchor_base.shape[0] * 2, 1, 1, 0)
        if TURN_ON_include_confidence:
            self.conf = nn.Conv2d(mid_channels, self.anchor_base.shape[0] * 2, 1, 1, 0) # Brad: 2025-02-05
        self.score = nn.Conv2d(mid_channels, self.anchor_base.shape[0] * num_score, 1, 1, 0) # BRAD: 2025-01-20
        self.proposal_creator = ProposalCreator(self, nms_thresh=nms_thresh, n_train_pre_nms=n_train_pre_nms, n_train_post_nms=n_train_post_nms, n_test_pre_nms=n_test_pre_nms, n_test_post_nms=n_test_post_nms, min_size=min_size)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

    def forward(self, features, img_size):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            features (~torch.autograd.Variable): The Features extracted from images. Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`, which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of proposal boxes.  This is a concatenation of bounding box arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted bounding boxes from the :math:`i` th image, :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. Its shape is :math:`(H W A, 4)`.
        """
        N, _, H, W = features.shape # Brad: (N, C_in, H, W)
        # n_anchor = anchor.shape[0] // (H * W) # A = n_anchor
        A = self.anchor_base.shape[0]
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, H, W) # (H*W*A, 4)

        h = F.relu(self.conv1(features)) # Brad: (N, C_mid, H, W)

        #--- location ------------------------------------
        rpn_locs = self.loc(h) # Brad: (N, C_mid, H, W) -> (N, A*4, H, W)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(N, -1, 4) # Brad: (N, A*4, H, W) -> (N, H, W, A*4) -> (N, H*W*A, 4)

        #--- confidence ----------------------------------
        if TURN_ON_include_confidence:
            rpn_confs = self.conf(h) # Brad: (N, C_mid, H, W) -> (N, A*1, H, W)
            rpn_confs = rpn_confs.permute(0, 2, 3, 1).contiguous().view(N, -1, 2) # Brad: (N, A*2, H, W) -> (N, H, W, A*2) -> (N, H*W*A, 2)
        else:
            rpn_confs = None

        #--- score ---------------------------------------
        rpn_scores = self.score(h) # Brad: (N, C_mid, H, W) -> (N, A*2, H, W)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() # Brad: (N, A*2, H, W) -> (N, H, W, A*2)
        # rpn_scores = rpn_scores.view(N, -1, 2) # Brad: (N, H, W, A*2) -> (N, H*W*A, 2)
        rpn_scores = rpn_scores.view(N, -1, self.num_score) # Brad: (N, H, W, A*2) -> (N, H*W*A, 2)

        return rpn_locs, rpn_confs, rpn_scores, anchor # (N, H*W*A, 4), (N, H*W*A, 2), (K, 4), (K,), (H*W*A, 4) # K <= N*H*W*A

# %%
class AnchorTargetCreator_BETA(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the sampled regions.

    """

    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5, use_original_subsample_for_postive_labels=True):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
        self.use_original_subsample_for_postive_labels = use_original_subsample_for_postive_labels

    # def __call__(self, bbox, anchor, img_size):
    def __call__(self, bbox, label_orig, anchor, img_size, cell_size_x=None, cell_size_y=None):
        """Assign ground truth supervision to sampled subset of anchors.
        Types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which is a tuple of height and width of an image.

        Returns:
            (array, array):
            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape is :math:`(S,)`.
        """

        IMG_H, IMG_W = img_size

        if TURN_ON_anchor_box_encoding:
            #===================================================================================
            assert cell_size_x is not None and cell_size_y is not None
            debug = False

            if debug:
                cell_size_x = 1
                cell_size_y = 1
                anchor = np.array([[0,0,1,1],[0.49,0.49,0.51,0.51], [1.1,0.1,1.2,0.2], [2,0,3,1],[2.1,0.1,2.2,0.2], [3,0,4,1]]) # (A,4)
                bbox = np.array([[0,0,1,1], [3,0,4,1], [2.4,0.4,2.6,0.6], [1.5,0.5,1.6,0.6]]) # (B,4)
                label_orig = np.array([10,20,30,40])

            #--- reconstruct cell coordinates from anchor ----------------------------
            cy = 0.5*(anchor[:,0] + anchor[:,2]) # (A,), center point
            cx = 0.5*(anchor[:,1] + anchor[:,3]) # (A,), center point

            cells = np.empty(anchor.shape) # (A,4), initialize
            cells[:,0] = cy - 0.5*cell_size_y # y0
            cells[:,1] = cx - 0.5*cell_size_x # x0
            cells[:,2] = cy + 0.5*cell_size_y # y1
            cells[:,3] = cx + 0.5*cell_size_x # x1

            #--- center point of bbox ----------------------------------------------
            bbcy = 0.5*(bbox[:,0] + bbox[:,2]) # (B,): bbox center = (y0 + y1) / 2
            bbcx = 0.5*(bbox[:,1] + bbox[:,3]) # (B,): bbox center = (x0 + x1) / 2

            is_inside = (cells[:,None,0] <= bbcy) & (bbcy <= cells[:,None,2]) & (cells[:,None,1] <= bbcx) & (bbcx <= cells[:,None,3]) # (A,1) and (B,) -> (A,B)
            if debug:
                print(f'is_inside = \n{is_inside}')

            #--- max ious ----------------------------------------------------------
            ious = bbox_iou(anchor, bbox) # (A, 4) x (B, 4) -> (A, B)
            if debug:
                print(f'ious = \n{ious}')

            ious[~is_inside] = 0 # only for the cell where the box center point inside it

            gt_max_ious = ious.max(axis=0)
            max_ious = ious.max(axis=1)

            is_max_ious = (ious == gt_max_ious)
            if debug:
                print(f'is_max_ious = \n{is_max_ious}')

            #--- is both inside the cell -------------------------------------------
            is_both = is_inside & is_max_ious
            if debug:
                print(f'is_both = \n{is_both}')
            assert is_both.any(axis=0).all(), f'Brad error: if you see this error, it means this algorithm needs a through investigation and a possibly upgrade...'

            argmax_both = is_both.argmax(axis=1) # (A,)
            any_both = is_both.any(axis=1) # (A,)

            #--- loc ---------------------------------------------------------------
            bbox_ = bbox[argmax_both]
            loc = bbox2loc_BETA(cells, anchor, bbox_) # compute bounding box regression targets
            loc[~any_both,:] = 0 # may have no effect, thanks to label == -1 

            if TURN_ON_include_confidence:
                conf = np.full((anchor.shape[0],), -1, dtype=np.int32) # -1 = don'tcare. See also how loss functions handle -1 indexes
                conf[max_ious < self.neg_iou_thresh] = 0 # assign negative labels first so that positive labels can clobber them
                conf[any_both] = 1 #  "+ 1", because 0 is reserved for background
            else:
                conf = None
            #--- label -------------------------------------------------------------
            label = np.full((anchor.shape[0],), -1, dtype=np.int32)
            if TURN_ON_include_confidence_SIMPLE:
                label[max_ious < self.neg_iou_thresh] = 0 # assign negative labels first so that positive labels can clobber them
            label[any_both] = label_orig[argmax_both][any_both] + 1 #  "+ 1", because 0 is reserved for background
            if debug:
                print(f'label = {label}')
            #===================================================================================
            return bbox_, loc, conf, label

        else:
            if TURN_ON_include_outer_boxes:
                inside_index = np.arange(anchor.shape[0])
            else:
                inside_index = _get_inside_index(anchor, IMG_H, IMG_W)

            anchor_ = anchor[inside_index]
            label_, argmax_ious = self._create_label(anchor_, bbox)
            loc_ = bbox2loc(anchor_, bbox[argmax_ious]) # compute bounding box regression targets

            label_orig_ = label_orig[argmax_ious] # BRAD: 2025-01-20
            label_orig_[label_ == -1] = -1 # BRAD: 2025-01-20
            label_orig_[label_ == 0] = 0 # BRAD: 2025-01-20
            label_orig_[label_ > 0] += 1 # BRAD: 2025-01-20

            # map up to original set of anchors
            # label = _unmap(label_, len(anchor), inside_index, fill=-1)
            label = _unmap(label_orig_, len(anchor), inside_index, fill=-1) # BRAD: 2025-01-20
            loc = _unmap(loc_, len(anchor), inside_index, fill=0)

            return loc, label

    def _create_label(self, anchor, bbox): # label: 1 is positive, 0 is negative, -1 is dont care
        argmax_ious, max_ious, _, _, gt_argmax_ious_all = _calc_ious(anchor, bbox)

        label = np.empty((anchor.shape[0],), dtype=np.int32)
        label.fill(-1) # initialize with 'dont care'

        label[max_ious < self.neg_iou_thresh] = 0 # assign negative labels first so that positive labels can clobber them
        label[gt_argmax_ious_all] = 1 # positive label: for each gt, anchor with highest iou
        if TURN_ON_anchor_selection_for_yolo:
            pass
        else:
            label[max_ious >= self.pos_iou_thresh] = 1 # positive label: above threshold IOU

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            if self.use_original_subsample_for_postive_labels:
                disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
                label[disable_index] = -1

            else: # Brad's algorithm
                i2i = np.empty((len(max_ious),), dtype=np.int32) # index-to-index vector
                i2i.fill(-1) # initialize
                i2i[pos_index] = range(len(pos_index))
                assert len(np.where(i2i[gt_argmax_ious_all] < 0)[0]) == 0 # make sure there is no '-1', which was used for initialization. Assume that gt_argmax_ious_all is a subset of pos_index
                # print(f'i2i = {i2i}')

                pos_ious = max_ious[pos_index].copy() # subgroup of max_ious
                pos_ious[i2i[gt_argmax_ious_all]] = 1.0 # forcefully assign the highest iou
                # print(f'pos_ious = {pos_ious}')

                disable_index = np.argsort(pos_ious)[:-n_pos] # pick the rest, outside top 'n_pos'
                label[pos_index[disable_index]] = -1 # assign 'dont care'
                # print(f'label = {label}')

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return label, argmax_ious

# if __name__=='__main__': # Brad
#     anchor_target_creator = AnchorTargetCreator()

#     anchor = np.array([[-1,-1,0,0], [0,0,1,1], [0.5,0.5,1.5,1.5], [1,1,2,2], [9,9,10,10]], dtype=np.float32)
#     bbox = np.array([[0.9,0.9,1.9,1.9], [0.2,0.2,1.2,1.2]], dtype=np.float32)
#     img_size = (10, 10)

#     loc, label = anchor_target_creator(bbox, anchor, img_size)
#     print(f'loc = {loc}')
#     print(f'label = {label}')


# %%
class YOLOv2(nn.Module):
    def __init__(
        self,
        n_fg_class=20, 
        feat_stride = 16,  # downsample 16x for output of conv5 in vgg16
        loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
        backbone_name='vgg16', use_pretrained_weights=True, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=False, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
        #--- parameters for region proposal module ---------------------------------
        base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None,
        nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16,   
        roi_size=7, is_roi_align=False, roi_align_sampling_ratio=-1, roi_align_aligned=False,
        #--- parameters for training model -----------------------------------------
        atc_n_sample=256, atc_pos_iou_thresh=0.7, atc_neg_iou_thresh=0.3, atc_pos_ratio=0.5, atc_use_original_subsample_for_postive_labels=True,
        ptc_n_sample=128, ptc_pos_ratio=0.25, ptc_pos_iou_thresh=0.5, ptc_neg_iou_thresh_hi=0.5, ptc_neg_iou_thresh_lo=0.0,
        ):

        super().__init__()
        self._init_inputs = {k:v for k,v in locals().items() if k not in ['self','__class__']} # save input arguments

        if backbone_name=='vgg16':
            extractor, classifier, in_channels, mid_channels, out_classifier = decom_vgg16(use_pretrained_weights=use_pretrained_weights, freeze_param_up_to=freeze_param_up_to, ceil_mode_for_MaxPool2d=ceil_mode_for_MaxPool2d)
        elif backbone_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d']:
            extractor, classifier, in_channels, mid_channels, out_classifier = decom_resnet(backbone_name=backbone_name, use_pretrained_weights=use_pretrained_weights, freeze_param_up_to=freeze_param_up_to, ceil_mode_for_MaxPool2d=ceil_mode_for_MaxPool2d, always_freeze_first_batch_norm=always_freeze_first_batch_norm)
        else:
            raise(Exception(f'Brad error: not available backbone name: {backbone_name}...'))

        # if use_untrained_vgg_classifier:
        #     classifier = [
        #         # nn.Linear(in_features=25088, out_features=4096, bias=True),
        #         nn.Linear(in_features=in_channels * (roi_size * roi_size), out_features=4096, bias=True), # Brad update: 2024-10-12
        #         nn.ReLU(inplace=True),
        #         nn.Linear(in_features=4096, out_features=4096, bias=True),
        #         nn.ReLU(inplace=True),
        #     ]
        #     classifier = [nn.Flatten()] + classifier

        #     classifier = nn.Sequential(*classifier)
        #     out_classifier = 4096
        #     print('Brad info: new classifier is built, instead of VGG16 classifier')

        num_score = n_fg_class + 1 # BRAD: 2025-01-20
        self.num_score = num_score # BRAD: 2025-01-20

        self.extractor = extractor
        # self.rpn = RegionProposalNetwork(in_channels, mid_channels, ratios=ratios, anchor_scales=anchor_scales, y_offsets=y_offsets, x_offsets=x_offsets, num_offsets=num_offsets, feat_stride=feat_stride, base_size=base_size, nms_thresh=nms_thresh, n_train_pre_nms=n_train_pre_nms, n_train_post_nms=n_train_post_nms, n_test_pre_nms=n_test_pre_nms, n_test_post_nms=n_test_post_nms, min_size=min_size)
        self.rpn = RegionProposalNetwork_BETA(in_channels, mid_channels, ratios=ratios, anchor_scales=anchor_scales, y_offsets=y_offsets, x_offsets=x_offsets, num_offsets=num_offsets, feat_stride=feat_stride, base_size=base_size, num_score=num_score, nms_thresh=nms_thresh, n_train_pre_nms=n_train_pre_nms, n_train_post_nms=n_train_post_nms, n_test_pre_nms=n_test_pre_nms, n_test_post_nms=n_test_post_nms, min_size=min_size) # BRAD: 2025-01-20
        # self.head = VGG16RoIHead(n_class=n_fg_class + 1, roi_size=roi_size, spatial_scale=(1. / feat_stride), classifier=classifier, out_classifier=out_classifier, is_roi_align=is_roi_align, roi_align_sampling_ratio=roi_align_sampling_ratio, roi_align_aligned=roi_align_aligned)
        self.n_class = n_fg_class + 1

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

        #--- parameters for training model -----------------------------------------
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        # self.anchor_target_creator = AnchorTargetCreator(n_sample=atc_n_sample, pos_iou_thresh=atc_pos_iou_thresh, neg_iou_thresh=atc_neg_iou_thresh, pos_ratio=atc_pos_ratio, use_original_subsample_for_postive_labels=atc_use_original_subsample_for_postive_labels)
        self.anchor_target_creator = AnchorTargetCreator_BETA(n_sample=atc_n_sample, pos_iou_thresh=atc_pos_iou_thresh, neg_iou_thresh=atc_neg_iou_thresh, pos_ratio=atc_pos_ratio, use_original_subsample_for_postive_labels=atc_use_original_subsample_for_postive_labels) # BRAD: 2025-01-20
        # self.proposal_target_creator = ProposalTargetCreator(n_sample=ptc_n_sample, pos_ratio=ptc_pos_ratio, pos_iou_thresh=ptc_pos_iou_thresh, neg_iou_thresh_hi=ptc_neg_iou_thresh_hi, neg_iou_thresh_lo=ptc_neg_iou_thresh_lo)

        self.optimizer = self.get_optimizer()

        # indicators for training status
        # self.rpn_cm = ConfusionMeter(2)
        self.rpn_cm = ConfusionMeter(num_score) # BRAD: 2025-01-20
        self.roi_cm = ConfusionMeter(self.n_class)
        self._losses = {k:[] for k in ['total_loss','rpn_loc_loss','rpn_cls_loss','roi_loc_loss','roi_cls_loss']}

        # https://github.com/pytorch/vision/issues/223
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/2
        # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # return appr -1~1 RGB
        self.pytorch_normalze = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if TURN_ON_anchor_box_encoding:
            self.cell_size_x = feat_stride / (num_offsets if num_offsets else len(x_offsets))
            self.cell_size_y = feat_stride / (num_offsets if num_offsets else len(y_offsets))

    # @property
    # def n_class(self):
    #     # Total number of classes including the background.
    #     return self.head.n_class

    def pre_process(self, imgs):
        assert imgs.shape[-1] == 3 # (N, H, W, C): Channel = 3
        imgs = imgs.permute(0, 3, 1, 2) # torch.tensor: (N, H, W, C) -> (N, C, H, W)
        imgs = imgs.float()
        imgs = imgs / 255.
        imgs = self.pytorch_normalze(imgs)
        return imgs

    def forward(self, imgs):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            imgs (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is :math:`(R',)`.
        """
        imgs = self.pre_process(imgs) # (N, H, W, C) -> (N, C, H, W)

        img_size = imgs.shape[2:] # _, _, H, W = imgs.shape

        features = self.extractor(imgs) # (N, C, H, W)
        rpn_locs, rpn_confs, rpn_scores, anchor = self.rpn(features, img_size) # (N, H*W*A, 4), (N, H*W*A, 2), (K, 4), (K,), (H*W*A, 4) # K <= N*H*W*A

        return rpn_locs, rpn_confs, rpn_scores, anchor, None # BRAD: 2025-01-20: (N, H*W*A, 4), (N, H*W*A, 2)

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, rpn_bbox, rpn_prob, rpn_conf_prob): # only for 1-batch: (H*W*A, 4), (H*W*A, class+1)
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            prob_l = rpn_prob[:, l]

            if TURN_ON_include_confidence:
                mask = (prob_l > self.score_thresh) & (rpn_conf_prob[:,1] > 0.5)
            else:
                mask = prob_l > self.score_thresh

            cls_bbox_l = rpn_bbox[mask]
            prob_l = prob_l[mask]

            keep = nms(cls_bbox_l, prob_l, self.nms_thresh) # Brad: https://pytorch.org/vision/main/generated/torchvision.ops.nms.html

            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @torch.no_grad
    def predict(self, imgs, visualize=False, visualize_score_thresh=None):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            if visualize_score_thresh is not None:
                self.score_thresh = visualize_score_thresh

        _, IMG_H, IMG_W, _ = imgs.shape

        bboxes = list()
        labels = list()
        scores = list()
        rpn_locs, rpn_confs, rpn_scores, anchor, _ = self(imgs)  # BRAD: 2025-01-20: (N, H*W*A, 4), (N, H*W*A, class+1), (H*W*A, 4)
        anchor = to_tensor(anchor)
        for i, (rpn_loc, rpn_score) in enumerate(zip(rpn_locs, rpn_scores)): # (H*W*A, 4), (H*W*A, class+1)
            if TURN_ON_anchor_box_encoding:
                rpn_bbox = loc2bbox_BETA(anchor, rpn_loc, cell_size_x=self.cell_size_x, cell_size_y=self.cell_size_y)
            else:
                rpn_bbox = loc2bbox(anchor, rpn_loc) # (H*W*A, 4)
                # clip bounding box
                rpn_bbox[:, 0::2] = (rpn_bbox[:, 0::2]).clamp(min=0, max=IMG_H) # y
                rpn_bbox[:, 1::2] = (rpn_bbox[:, 1::2]).clamp(min=0, max=IMG_W) # x

            if TURN_ON_include_confidence:
                rpn_conf = rpn_confs[i]
                rpn_conf_prob = F.softmax(to_tensor(rpn_conf), dim=1) # (H*W*A, class+1)
            else:
                rpn_conf_prob = None

            rpn_prob = F.softmax(to_tensor(rpn_score), dim=1) # (H*W*A, class+1)
            bbox, label, score = self._suppress(rpn_bbox, rpn_prob, rpn_conf_prob) # only for 1-batch: (H*W*A, 4), (H*W*A, class+1)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    #--- for training model -----------------------------------------------------------
    def forward_to_train_model(self, imgs, bboxes, labels):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes. Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels. Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value is :math:`[0, L - 1]`. :math:`L` is the number of foreground classes.
            scale (float): Amount of scaling applied to the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        imgs = self.pre_process(imgs) # (N, H, W, C) -> (N, C, H, W)

        img_size = imgs.shape[2:] # _, _, H, W = imgs.shape

        features = self.extractor(imgs) # (N, C, H, W)
        rpn_locs, rpn_confs, rpn_scores, anchor = self.rpn(features, img_size) # (N, H*W*A, 4), (N, H*W*A, 2), (K, 4), (K,), (H*W*A, 4) # K <= N*H*W*A

        # ------------------ RPN losses -------------------#
        rpn_loc = rpn_locs.view(-1, 4) # (N, H*W*A, 4) -> (N*H*W*A, 4)
        # rpn_score = rpn_scores.view(-1, 2) # (N, H*W*A, 2) -> (N*H*W*A, 2)
        if TURN_ON_include_confidence:
            rpn_conf = rpn_confs.view(-1, 2) # (N, H*W*A, 2) -> (N*H*W*A, 2)
        rpn_score = rpn_scores.view(-1, self.num_score) # (N, H*W*A, 2) -> (N*H*W*A, 2) # BRAD: 2025-01-20

        gt_rpn_bbox, gt_rpn_loc, gt_rpn_conf, gt_rpn_label, anchors = [], [], [], [], []
        # for n, bbox in enumerate(bboxes):
        for n, (bbox, label) in enumerate(zip(bboxes, labels)): # BRAD: 2025-01-20
            if TURN_ON_anchor_box_encoding:
                gt_rpn_bbox_, gt_rpn_loc_, gt_rpn_conf_, gt_rpn_label_ = self.anchor_target_creator(to_numpy(bbox), to_numpy(label), anchor, img_size, cell_size_x=self.cell_size_x, cell_size_y=self.cell_size_y) # (H*W*A, 4), (H*W*A,)
            else:
                # gt_rpn_loc_, gt_rpn_label_ = self.anchor_target_creator(to_numpy(bbox), anchor, img_size) # (H*W*A, 4), (H*W*A,)
                gt_rpn_loc_, gt_rpn_label_ = self.anchor_target_creator(to_numpy(bbox), to_numpy(label), anchor, img_size) # (H*W*A, 4), (H*W*A,)
            gt_rpn_bbox.append(gt_rpn_bbox_)
            gt_rpn_loc.append(gt_rpn_loc_)
            if TURN_ON_include_confidence:
                gt_rpn_conf.append(gt_rpn_conf_)
            gt_rpn_label.append(gt_rpn_label_)
            anchors.append(anchor)
        gt_rpn_bbox = np.concatenate(gt_rpn_bbox, axis=0)
        gt_rpn_loc = np.concatenate(gt_rpn_loc, axis=0)
        if TURN_ON_include_confidence:
            gt_rpn_conf = np.concatenate(gt_rpn_conf, axis=0)
        gt_rpn_label = np.concatenate(gt_rpn_label, axis=0)
        anchors = np.concatenate(anchors, axis=0)

        if TURN_ON_include_confidence:
            gt_rpn_bbox, gt_rpn_loc, gt_rpn_conf, gt_rpn_label = to_tensor(gt_rpn_bbox), to_tensor(gt_rpn_loc), to_tensor(gt_rpn_conf).long(), to_tensor(gt_rpn_label).long()
        else:
            gt_rpn_loc, gt_rpn_label = to_tensor(gt_rpn_loc), to_tensor(gt_rpn_label).long()

        if TURN_ON_loss_bbox_as_is:
            rpn_loc_loss = _fast_rcnn_loc_loss_BETA2(rpn_loc, gt_rpn_loc, gt_rpn_label.data, to_tensor(anchors), self.cell_size_x, self.cell_size_y)
        else:
            rpn_loc_loss = _fast_rcnn_loc_loss_BETA(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma) # (H*W*A, 4) <-> (H*W*A, 4)
        if TURN_ON_include_confidence:
            if TURN_ON_include_confidence_split_conf:
                gt_rpn_conf_0 = torch.full(gt_rpn_conf.shape, -1, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
                gt_rpn_conf_0[gt_rpn_conf == 0] = 1
                gt_rpn_conf_1 = torch.full(gt_rpn_conf.shape, -1, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
                gt_rpn_conf_1[gt_rpn_conf == 1] = 1

                rpn_cnf_loss_0 = F.cross_entropy(rpn_conf, gt_rpn_conf_0, ignore_index=-1) # No object
                rpn_cnf_loss_1 = F.cross_entropy(rpn_conf, gt_rpn_conf_1, ignore_index=-1)
            else:
                rpn_cnf_loss_0 = F.cross_entropy(rpn_conf, gt_rpn_conf, ignore_index=-1)
                rpn_cnf_loss_1 = torch.tensor(0)
        else:
            rpn_cnf_loss_0, rpn_cnf_loss_1 = torch.tensor(0), torch.tensor(0)

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1) # NOTE: default value of ignore_index is -100 ...: (H*W*A, 2) <-> (H*W*A,)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = to_numpy(rpn_score)[to_numpy(gt_rpn_label) > -1]
        self.rpn_cm.add(to_tensor(_rpn_score, False), _gt_rpn_label.data.long())

        return {
            'total_loss': LAMBDA_COORD*rpn_loc_loss + rpn_cls_loss + LAMBDA_NOOBJ*rpn_cnf_loss_0 + rpn_cnf_loss_1,
            'rpn_loc_loss':rpn_loc_loss,
            'rpn_cls_loss':rpn_cls_loss,
            'roi_loc_loss':rpn_cnf_loss_0,
            'roi_cls_loss':rpn_cnf_loss_1,
        }


    def train_step(self, imgs, bboxes, labels, max_norm_of_clip_grad_norm=None):
        self.optimizer.zero_grad()
        out = self.forward_to_train_model(imgs, bboxes, labels)
        out['total_loss'].backward()
        if TURN_ON_gradient_clip_by_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=TURN_ON_gradient_clip_by_norm)
        self.optimizer.step()
        self.update_meters(out)
        return out

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.state_dict()
        save_dict['model_init_inputs'] = self._init_inputs # Brad: 2024-12-30
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        # save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        if 'model' in state_dict:
            # self.__init__(**state_dict['model_init_inputs']) # Brad: 2024-12-30
            self.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.load_state_dict(state_dict)
            return None
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return None

    def update_meters(self, out):
        for k,v in out.items(): # Brad: 2024-12-14
            if k in self._losses: # only in the list of pre-defined losses
                self._losses.setdefault(k, []).append(v.item()) # NOTE: '.item()' needs to be included, I think

    def reset_meters(self):
        self._losses = {k:[] for k in self._losses.keys()} # reset losses, Brad: 2024-12-14
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k:(statistics.mean(vv) if vv else float('nan')) for k, vv in self._losses.items()} # Brad: 2024-12-14


# %% [markdown]
# # Brad's OCR Dataset

# %%
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import pandas as pd
if __name__=='__main__':
    from bradk.datasets.advanced_texts import generate_random_sample

def pre_process_image_bboxes(img, bboxes=None, fix_img_H=None):
    if fix_img_H: # resize image and bbox
        H, W, C = img.shape
        img = cv2.resize(img.astype(np.float32), (math.ceil(W * fix_img_H / H), math.ceil(fix_img_H)), interpolation=(cv2.INTER_AREA if (fix_img_H / H) < 1 else cv2.INTER_LINEAR))
        if bboxes is not None:
            o_H, o_W, _ = img.shape
            bboxes = resize_bbox(bboxes, (H, W), (o_H, o_W)) # Brad: also resize bbox according to the resized (or preprocessed) image

    return img, bboxes

class AdvancedTextsDataset_YXYX(torch.utils.data.Dataset):
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

# %% [markdown]
# # Train

# %%
def eval_model(dataloader, obj_detector, test_num=10000, device=None):
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()

    for ii, (imgs, gt_bboxes_, gt_labels_, gt_difficults_) in (enumerate(custom_progressbar(dataloader)) if TURN_ON_PROGRESS_BAR else enumerate(dataloader)): # Brad 2024-06-22
        imgs = imgs.to(device).float()
        pred_bboxes_, pred_labels_, pred_scores_ = obj_detector.predict(imgs)

        gt_bboxes += [g.numpy() for g in gt_bboxes_]
        gt_labels += [g.numpy() for g in gt_labels_]
        gt_difficults += [g.numpy() for g in gt_difficults_]
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

        if ii == test_num: 
            break

    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
        # use_07_metric=True)
        use_07_metric=None)

    return result

# %% [markdown]
# - set up

# %%
if __name__=='__main__':
    from sklearn import metrics

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = xm.xla_device() # TPU
    print(f'device = {device}')

    if device == 'cpu':
        kwargs = {'env':'fasterrcnn', 'epoch':1 , 'num_workers':0, 'test_num_workers':0, 'voc_data_dir':r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007'}
        # kwargs = {'env':'fasterrcnn', 'epoch':0 , 'num_workers':0, 'test_num_workers':0, 'voc_data_dir':r'C:\Users\bomso\bomsoo1\python\_pytorch\data\voc2007\VOCdevkit\VOC2007'}
        # kwargs = {'env':'fasterrcnn', 'epoch':1 , 'num_workers':4, 'test_num_workers':4, 'voc_data_dir':r'/kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007'}
    else:
        # kwargs = {'env':'fasterrcnn', 'epoch':14, 'num_workers':4, 'test_num_workers':4, 'voc_data_dir':r'/kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007'}
        kwargs = {'env':'fasterrcnn', 'epoch':9, 'num_workers':4, 'test_num_workers':4, 'voc_data_dir':r'/kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007'}
        # kwargs = {'env':'fasterrcnn', 'epoch':1, 'num_workers':4, 'test_num_workers':4, 'voc_data_dir':r'/kaggle/input/datasets-voc-all-2007/VOCdevkit/VOC2007'}
    print(kwargs)

    opt._parse(kwargs)

    #------------------------------------------------------------------------
    print('load data')
    if False: # VOC dataset
        trainset = VOCDataset(opt, split='trainval', use_difficult=False) # Brad 2024-06-22
        testset = VOCDataset(opt, split='test', use_difficult=True) # Brad 2024-06-22

        model_inputs = dict(
            n_fg_class=20, 
            feat_stride = 16,  # downsample 16x for output of conv5 in vgg16
            ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None,
            # loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            backbone_name='vgg16', use_pretrained_weights=True, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=False, use_untrained_vgg_classifier=False,
            # backbone_name='resnet101', use_pretrained_weights=True, freeze_param_up_to=None, ceil_mode_for_MaxPool2d=False, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
            #--- region proposal ---------------------------
            base_size=16,
            # nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16,
            # roi_size=7, is_roi_align=False, roi_align_sampling_ratio=-1, roi_align_aligned=False, 
            #--- parameters for training model ----------------------------------
            atc_n_sample=256, atc_pos_iou_thresh=0.7, atc_neg_iou_thresh=0.3, atc_pos_ratio=0.5,
            ptc_n_sample=128, ptc_pos_ratio=0.25, ptc_pos_iou_thresh=0.5, ptc_neg_iou_thresh_hi=0.5, ptc_neg_iou_thresh_lo=0.0,
        )

        label_names = trainset.db.label_names

    else: # Brad's OCR dataset
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

        if TURN_ON_YOLOv2: # 1 character for test
            dataset_inputs = dict(
                # object_type='word', # WORD APPLICATION
                object_type='character', # CHARACTER APPLICATION
                # fix_img_H = None, # WORD APPLICATION / CHARACTER APPLICATION
                # fix_img_H = 100, # CHARACTER APPLICATION 2
                fix_img_H = 30, # CHARACTER APPLICATION 2
                #--- sample generation from hard drive ------------------------
                dirpaths=[], subdir_imgs='images', subidr_segs='segmentations', filename_annotation='annotation.json',
                #--- sample generation in real time ---------------------------
                # num_real_time_samples=1,
                # num_real_time_samples=500,
                num_real_time_samples=5000,

                # font_filepath_list=[
                #     "arial.ttf" # for desktop use
                #     if device == 'cpu' else
                #     r'/kaggle/input/dataset-text-scene/fonts_321/fonts_321/ARIALN.TTF' # for Kaggle
                #     ],
                font_filepath_list=get_font_filepath(
                    r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\fonts_321' # for desktop use
                    if device == 'cpu' else
                    r'/kaggle/input/dataset-text-scene/fonts_321/fonts_321' # for Kaggle
                    ), 

                # characters = list('''0123456789'''), # ONE CHAR
                characters = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+-={}|[]\:";'<>?,./'''), # full except: `

                word_range = range(1,12),
                # word_range = range(1,2), # ONE CHAR

                num_words_range = range(1,2), # CHARACTER APPLICATION
                num_lines_range = range(1,2), # CHARACTER APPLICATION  
                indent_range = range(0,1), # CHARACTER APPLICATION

                # generate_OCR_image
                # font_size_range = range(30, 31), # TEST
                font_size_range = range(10, 80),

                # font_size_weights = None,
                font_size_weights = [1/i for i in range(10, 80)],

                # line_spacing_range = range(0, 1),
                line_spacing_range = range(0, 5),

                # angle_range=[0],
                # img_size_xy=(90, 90), img_size_dx_range=None, img_size_dy_range=None, orig_point_max_ratio_range = 0.66, # WORD APPLICATION
                img_size_xy = None, img_size_dx_range=None, img_size_dy_range=None, crop_for_only_characters=True, # CHARACTER APPLICATION 2
                # prob_draw_top_line = 0, # [0, 1]
                # prob_draw_bottom_line = 0, # [0, 1]
                # prob_draw_left_line = 0, # [0, 1]
                # prob_draw_right_line = 0, # [0, 1]
                # prob_draw_inner_line_yoffset = 0, # [0, 1]                
                )

            trainset = AdvancedTextsDataset_YXYX(**{**dataset_inputs, 'dirpaths':[
                #     r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\train',
                # ] if device == 'cpu' else [
                #     r'/kaggle/input/dataset-text-scene/datasets_NONE_FONT10_20_BATCH2500/train',
                ]})
            testset = AdvancedTextsDataset_YXYX(**{**dataset_inputs, 'dirpaths':[ # Brad 2024-06-22
                    r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\test',
                ] if device == 'cpu' else [
                    # r'/kaggle/input/dataset-text-scene/datasets_ver2_fullexcept_600x600_FONT10_80_BATCH1000/test', # WORD APPLICATION
                    # r'/kaggle/input/dataset-text-scene/datasets_ver3_fullexcept_600ax600a_FONT10_80_BATCH1000/test', # WORD APPLICATION 2
                    r'/kaggle/input/dataset-text-scene/datasets_ver2_fullexcept_NONE_FONT10_80_BATCH1000/test', # CHARACTER APPLICATION
                ]})

        else:
            dataset_inputs = dict(
                object_type='word', # WORD APPLICATION
                # object_type='character', # CHARACTER APPLICATION

                fix_img_H = None, # WORD APPLICATION / CHARACTER APPLICATION
                # fix_img_H = 120, # CHARACTER APPLICATION
                # fix_img_H = 100, # CHARACTER APPLICATION 2

                #--- sample generation from hard drive ------------------------
                dirpaths=[], subdir_imgs='images', subidr_segs='segmentations', filename_annotation='annotation.json',

                #--- sample generation in real time ---------------------------
                # num_real_time_samples=1,
                # num_real_time_samples=500,
                num_real_time_samples=5000,

                font_filepath_list=get_font_filepath(
                    r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\fonts_321' # for desktop use
                    if device == 'cpu' else
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
                #     if device == 'cpu' else
                #     # r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.3,
                #     r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.5,
                #     # r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.7,
                #     # r'/kaggle/input/dataset-text-scene/words.txt'), list_words_prob = 0.9,

                num_words_range = range(0,10), # WORD APPLICATION
                # num_words_range = range(5,15), # WORD APPLICATION 2
                # num_words_range = range(1,2), # CHARACTER APPLICATION
                num_lines_range = range(1,20), # WORD APPLICATION
                # num_lines_range = range(10,30), # WORD APPLICATION 2
                # num_lines_range = range(1,2), # CHARACTER APPLICATION
                indent_range = range(0,15), # WORD APPLICATION
                # indent_range = range(0,3), # WORD APPLICATION 2
                # indent_range = range(0,1), # CHARACTER APPLICATION

                # generate_OCR_image
                # font_size_range = range(10, 20), # TEST
                # font_size_range = range(20, 40), # TEST
                # font_size_range = range(40, 60), # TEST
                # font_size_range = range(60, 80), # TEST
                font_size_range = range(10, 80),

                # font_size_weights = None,
                font_size_weights = [1/i for i in range(10, 80)],

                line_spacing_range = range(0, 5),

                img_size_xy=(600, 600), img_size_dx_range=None, img_size_dy_range=None, orig_point_max_ratio_range = 0.5, # WORD APPLICATION
                # img_size_xy=(600, 600), img_size_dx_range=range(0,200), img_size_dy_range=range(0,200), orig_point_max_ratio_range = 0.1, # WORD APPLICATION 2
                # img_size_xy = None, img_size_dx_range=None, img_size_dy_range=None, crop_for_only_characters=False, # CHARACTER APPLICATION
                # img_size_xy = None, img_size_dx_range=None, img_size_dy_range=None, crop_for_only_characters=True, # CHARACTER APPLICATION 2

                )

            trainset = AdvancedTextsDataset_YXYX(**{**dataset_inputs, 'dirpaths':[
                #     r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\train',
                # ] if device == 'cpu' else [
                #     r'/kaggle/input/dataset-text-scene/datasets_NONE_FONT10_20_BATCH2500/train',
                ]})
            testset = AdvancedTextsDataset_YXYX(**{**dataset_inputs, 'dirpaths':[ # Brad 2024-06-22
                    r'C:\Users\bomso\bomsoo1\python\bradk\bradk\datasets\test',
                ] if device == 'cpu' else [
                    r'/kaggle/input/dataset-text-scene/datasets_ver2_fullexcept_600x600_FONT10_80_BATCH1000/test', # WORD APPLICATION
                    # r'/kaggle/input/dataset-text-scene/datasets_ver3_fullexcept_600ax600a_FONT10_80_BATCH1000/test', # WORD APPLICATION 2
                    # r'/kaggle/input/dataset-text-scene/datasets_ver2_fullexcept_NONE_FONT10_80_BATCH1000/test', # CHARACTER APPLICATION
                ]})

        model_inputs = dict(
            n_fg_class=len(trainset.label_names), 
            feat_stride = 16,  # downsample 16x for output of conv5 in vgg16

            base_size=16,
            # ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], y_offsets=[0], x_offsets=[0], num_offsets=None,
            # ratios=[0.25, 0.5, 1, 2, 4], anchor_scales=[1, 2, 4], y_offsets=[0], x_offsets=[0], num_offsets=None,
            ratios=[0.25, 0.5, 1, 2, 4], anchor_scales=[0.25, 0.5, 1, 2, 4], y_offsets=[0], x_offsets=[0], num_offsets=None, # WORD APPLICATION
            # ratios=[0.25, 0.5, 1, 2, 4], anchor_scales=[0.25, 0.5, 1, 2, 4], y_offsets=[-16/3*1, 0, 16/3*1], x_offsets=[-16/3*1, 0, 16/3*1], num_offsets=None, # CHARACTER APPLICATION
            # ratios=[1], anchor_scales=[0.0625], y_offsets=[-16/7*3, -16/7*2, -16/7*1, 0, 16/7*1, 16/7*2, 16/7*3], x_offsets=[-16/7*3, -16/7*2, -16/7*1, 0, 16/7*1, 16/7*2, 16/7*3], num_offsets=None,

            # loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2),

            # backbone_name='vgg16', use_pretrained_weights=True, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=False, use_untrained_vgg_classifier=False,
            # backbone_name='vgg16', use_pretrained_weights=True, freeze_param_up_to=10, ceil_mode_for_MaxPool2d=True, use_untrained_vgg_classifier=False,
            backbone_name='vgg16', use_pretrained_weights=True, freeze_param_up_to=0, ceil_mode_for_MaxPool2d=True, use_untrained_vgg_classifier=False,
            # backbone_name='resnet101', use_pretrained_weights=True, freeze_param_up_to=None, ceil_mode_for_MaxPool2d=False, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
            # backbone_name='resnet101', use_pretrained_weights=True, freeze_param_up_to=5, ceil_mode_for_MaxPool2d=False, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
            # backbone_name='resnet101', use_pretrained_weights=True, freeze_param_up_to=0, ceil_mode_for_MaxPool2d=False, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
            # backbone_name='resnet101', use_pretrained_weights=True, freeze_param_up_to=0, ceil_mode_for_MaxPool2d=True, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,
            # backbone_name='resnext101_32x8d', use_pretrained_weights=True, freeze_param_up_to=0, ceil_mode_for_MaxPool2d=True, always_freeze_first_batch_norm=False, use_untrained_vgg_classifier=False,

            #--- region proposal ---------------------------
            # nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16,
            # nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=0, # min_size --> small object detection
            nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=12000, n_test_post_nms=2000, min_size=0, # min_size --> small object detection

            # roi_size=7, is_roi_align=False, roi_align_sampling_ratio=-1, roi_align_aligned=False, # ROIPool
            roi_size=7, is_roi_align=True, roi_align_sampling_ratio=2, roi_align_aligned=True, # ROIAlign

            #--- parameters for training model ----------------------------------
            atc_n_sample=256, atc_pos_iou_thresh=0.7, atc_neg_iou_thresh=0.3, atc_pos_ratio=0.5, atc_use_original_subsample_for_postive_labels=True,
            # atc_n_sample=256, atc_pos_iou_thresh=0.7, atc_neg_iou_thresh=0.5, atc_pos_ratio=0.5, atc_use_original_subsample_for_postive_labels=True,
            # atc_n_sample=256, atc_pos_iou_thresh=0.0001, atc_neg_iou_thresh=0.0001, atc_pos_ratio=0.5, atc_use_original_subsample_for_postive_labels=True,

            ptc_n_sample=128, ptc_pos_ratio=0.25, ptc_pos_iou_thresh=0.5, ptc_neg_iou_thresh_hi=0.5, ptc_neg_iou_thresh_lo=0.0,
            ) 

        label_names = trainset.label_names

    ###############################################################
    def custom_collate_fn(batch): # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2
        img_BATCH, bboxes_BATCH, labels_BATCH, difficult_BATCH = [], [], [], []
        for img, bboxes, labels, difficult in batch:
            img_BATCH.append(img)
            bboxes_BATCH.append(torch.tensor(bboxes))
            labels_BATCH.append(torch.tensor(labels))
            difficult_BATCH.append(torch.tensor(difficult))
        img_BATCH = torch.tensor(np.stack(img_BATCH)) # assume that all images are of the same shape
        return img_BATCH, bboxes_BATCH, labels_BATCH, difficult_BATCH

    # train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn, pin_memory=False)
    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn, pin_memory=True)
    test_dataloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=opt.test_num_workers, collate_fn=custom_collate_fn, pin_memory=True)

    #-------------------------------------------------------------
    if TURN_ON_YOLOv2: # BRAD: 2025-01-20
        obj_detector = YOLOv2(**model_inputs) # BRAD: 2025-01-20
    else:
        obj_detector = FasterRCNN(**model_inputs)
    obj_detector.to(device)
    print('model construct completed')

    #-------------------------------------------------------------
    if opt.load_path:
        obj_detector.load(opt.load_path, map_location=torch.device(device))
        print('load pretrained model from %s' % opt.load_path)

    best_map = 0
    lr_ = opt.lr

# %% [markdown]
# - load pre-trained model

# %%
if __name__=='__main__':
    if False:
        d = r'/kaggle/input/pjt-faster-rcnn-20240622/checkpoints'
        all_trained = sorted([os.path.join(d,f) for f in os.listdir(d)])
        print(all_trained)
        filepath_trained = all_trained[-1]
        print(f'filepath_trained = {filepath_trained}')

        state_dict_ = torch.load(filepath_trained, map_location=torch.device(device))
        # obj_detector.__init__(**state_dict['model_init_inputs']) # Brad: 2024-12-30
        obj_detector.load_state_dict(state_dict_['model']) # overwrite

        history = json.load(open(r'/kaggle/input/pjt-faster-rcnn-20240622/history.json', 'r'))
    else:
        history = {}

        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

# %% [markdown]
# - main train loop

# %%
if __name__=='__main__':
    for epoch in range(opt.epoch):
        obj_detector.reset_meters()

        for ii, (img, bbox_, label_, _) in (enumerate(custom_progressbar(train_dataloader)) if TURN_ON_PROGRESS_BAR else enumerate(train_dataloader)): # Brad 2024-06-22
            img = img.to(device).float()
            bbox = [b.to(device) for b in bbox_] # list of tensors, see custom_collate_fn
            label = [l.to(device) for l in label_] # list of tensors, see custom_collate_fn

            obj_detector.train_step(img, bbox, label, max_norm_of_clip_grad_norm=None)

        #=================================================================================
        #--- plot ground truth & predicted boxes (training dataset) ----------------------
        _bboxes, _labels, _scores = obj_detector.predict(img, visualize=True)

        ori_img_ = to_numpy(img[0])

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,20)) # Brad 
        vis_bbox(ori_img_, to_numpy(bbox_[0]), to_numpy(label_[0]), ax=ax1, label_names=label_names) # plot groud truth bboxes
        vis_bbox(ori_img_, to_numpy(_bboxes[0]), to_numpy(_labels[0]).reshape(-1), to_numpy(_scores[0]), ax=ax2, label_names=label_names) # plot predicted bboxes
        plt.show() # Brad

        #--- plot confusion matrix (training dataset)  ----------------------------------
        if TURN_ON_YOLOv2: # BRAD: 2025-01-20
            confusion_matrix = obj_detector.rpn_cm.value() # BRAD: 2025-01-20
        else:
            confusion_matrix = obj_detector.roi_cm.value() # Brad
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=range(obj_detector.n_class)) # Brad

        fig, ax = plt.subplots(figsize=(12,12))
        cm_display.plot(ax=ax)
        plt.show()

        #--- evaluate with test dataset ------------------------------------------------
        eval_result = eval_model(test_dataloader, obj_detector, test_num=opt.test_num, device=device)
        lr_ = obj_detector.optimizer.param_groups[0]['lr']

        #--- save/display metrics -------------------------------
        history.setdefault('epoch',[]).append(epoch)
        history.setdefault('lr',[]).append(lr_)
        for k,v in {**eval_result, **obj_detector.get_meter_data()}.items():
            if not isinstance(v, (list, tuple, set, dict, np.ndarray, torch.Tensor)): # save only single values
                history.setdefault(k,[]).append(v)
        print(', '.join([f'{k} = {v[-1]}' for k,v in history.items()]) + f", time = {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        json.dump(history, open('history.json', 'w'))

        #--- save model -------------------------------------------
        if eval_result['mAP'] > best_map:
            best_map = eval_result['mAP']
            best_path = obj_detector.save(best_map=best_map)

        #--- adjust learning rate parameter -----------------------
        if epoch == 9:
            obj_detector.load(best_path, map_location=torch.device(device))
            obj_detector.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

# %% [markdown]
# # Plot History

# %%
if __name__=='__main__':
    import pandas as pd
    pd.set_option('display.max_rows', 500)

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    df[['mAP_07','mAP']].plot(ax=axes[0], figsize=(12,4), logy=False)
    df[['total_loss','rpn_loc_loss','rpn_cls_loss','roi_loc_loss','roi_cls_loss']].plot(ax=axes[1], figsize=(12,4), logy=False)
    plt.show()

    display(df)

# %% [markdown]
# # Load Pre-trained Model

# %%
class BradOCR:
    def __init__(
            self, 
            filepath_model_word=None, filepath_model_char=None, n_test_pre_nms=12000, n_test_post_nms=2000,
            label_names_char = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+-={}|[]\:";'<>?,./'''), # full except: `
            ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #--- load models ----------------------------------
        reloaded_word = torch.load(filepath_model_word, map_location=torch.device(device))
        reloaded_word['model_init_inputs']['n_test_pre_nms'] = n_test_pre_nms # modify model inputs
        reloaded_word['model_init_inputs']['n_test_post_nms'] = n_test_post_nms # modify model inputs
        print(f">> reloaded_word['model_init_inputs'] = {reloaded_word['model_init_inputs']}")

        reloaded_char = torch.load(filepath_model_char, map_location=torch.device(device))
        reloaded_char['model_init_inputs']['n_test_pre_nms'] = n_test_pre_nms # modify model inputs
        reloaded_char['model_init_inputs']['n_test_post_nms'] = n_test_post_nms # modify model inputs
        print(f">> reloaded_char['model_init_inputs'] = {reloaded_char['model_init_inputs']}")

        #--- load model ----------------------------------
        self.models = {}

        self.models['word'] = FasterRCNN(**reloaded_word['model_init_inputs'])
        self.models['word'].load_state_dict(reloaded_word['model']) # full except: `

        self.models['char'] = FasterRCNN(**reloaded_char['model_init_inputs'])
        self.models['char'].load_state_dict(reloaded_char['model']) # full except: `

        self.label_names_char = label_names_char

    def predict(self, img, fix_img_H=100, debug=False):
        #--- predict words -----------------------------------------------------------------
        pred_bboxes, pred_labels, pred_scores = self.models['word'].predict(to_tensor(img[None]), visualize=True) # Brad decision 7/6/2024

        if debug:
            vis_bbox(img, to_numpy(pred_bboxes[0]), label=None, score=None, ax=None, label_names=None, linewidth=0.5, figsize=(20,20)) # plot groud truth bboxes
            plt.show() # Brad

        #--- predict characters -----------------------------------------------------------
        pred_texts = []
        for y0, x0, y1, x1 in custom_progressbar(pred_bboxes[0]):
            y0, x0, y1, x1 = math.floor(y0), math.floor(x0), math.ceil(y1), math.ceil(x1)

            #--- crop image to find text region -------------------------------------------
            img_text = img[y0:(y1+1), x0:(x1+1), :].copy() # crop image to extract target text
            img_text, _ = pre_process_image_bboxes(img_text, bboxes=None, fix_img_H=fix_img_H) # resize text image

            #--- predict characters on text image -----------------------------------------
            pred_bboxes_char, pred_labels_char, pred_scores_char = self.models['char'].predict(to_tensor(img_text[None]), visualize=True, visualize_score_thresh=0.05) # Brad decision 7/6/2024

            #--- nms ---------------------------------------------------------------------
            ikeep = nms(torch.from_numpy(pred_bboxes_char[0]), torch.from_numpy(pred_scores_char[0]), 0.25)
            pred_bbox_, pred_label_, pred_score_ = pred_bboxes_char[0][ikeep], pred_labels_char[0][ikeep], pred_scores_char[0][ikeep]
            ii = pred_score_ > 0.4
            pred_bbox, pred_label, pred_score = pred_bbox_[ii], pred_label_[ii], pred_score_[ii]
            isort = np.argsort(pred_bbox[:,1] + pred_bbox[:,3])
            pred_bbox, pred_label, pred_score = pred_bbox[isort], pred_label[isort], pred_score[isort]

            pred_texts.append((
                ''.join(self.label_names_char[i] for i in pred_label),
                (x0, y0, x1, y1),
                ))

            if debug:
                # print(pred_texts[-1])
                vis_bbox(img_text, pred_bbox, label=pred_label.reshape(-1), score=pred_score, ax=None, label_names=self.label_names_char, linewidth=0.5) # plot groud truth bboxes
                plt.show() # Brad
                # break

        return {
            'pred_texts':pred_texts,
        }

# %%
def plot_text_xyxy(img, pred_texts):
    pred_bboxes = [(y0,x0,y1,x1) for t,(x0,y0,x1,y1) in pred_texts] # convert from xyxy to yxyx

    vis_bbox(
        img, np.array(pred_bboxes), 
        label=np.array(list(range(len(pred_texts)))), 
        # score=pred_scores[0], 
        ax=None, 
        # label_names=[re.escape(t) for t in pred_texts], 
        label_names=[''.join({'$':'\$','^':'\^','_':'\_'}.get(c,c) for c in t) for t, (x0, y0, x1, y1) in pred_texts], 
        linewidth=0.5, 
        figsize=(20,20), fontsize=7,
        )
    plt.show()
    return

# if __name__=='__main__':
    # pred_texts = out['pred_texts']
    # plot_text_xyxy(pred_texts)

# %%
# if __name__=='__main__':
#     dirpath = r'C:\Users\bomso\bomsoo1\python\_pytorch\pjt_faster_rcnn'
#     ocr_model = BradOCR(
#         #-------------------------------------------------------
#         filepath_model_word=os.path.join(dirpath, r'trained/word/fasterrcnn_01010320_0.9651181492864416'), # resnet
#         # filepath_model_word=r'./trained/word/fasterrcnn_01091011_0.9400209864454363', # vgg, WORD2
#         #-------------------------------------------------------
#         # filepath_model_char=r'./trained/char/fasterrcnn_01051628_0.9574360212093982',  # vgg
#         # filepath_model_char=r'./trained/char/fasterrcnn_01052355_0.9604068342185424',  # resnet
#         # filepath_model_char=r'./trained/char/fasterrcnn_01182318_0.9560645224748588',  # faster_rcnn_250118_OCR_CHAR2_H100 vgg
#         # filepath_model_char=r'./trained/char/fasterrcnn_01190224_0.9659664098008948',  # faster_rcnn_250118_OCR_CHAR2_H100 vgg
#         # filepath_model_char=os.path.join(dirpath, r'trained/char/fasterrcnn_01190457_0.9713410156546154'),  # faster_rcnn_250118_OCR_CHAR2_H100 vgg

#         # filepath_model_char=os.path.join(dirpath, r'trained/char/fasterrcnn_01270140_0.9694273477337261'),  # pjt_faster_rcnn_250126_OCR_CHAR3_0p3 vgg
#         filepath_model_char=os.path.join(dirpath, r'trained/char/fasterrcnn_01270246_0.9668653988020294'),  # pjt_faster_rcnn_250126_OCR_CHAR3_0p5 vgg
#         # filepath_model_char=os.path.join(dirpath, r'trained/char/fasterrcnn_01270311_0.9613085096333355'),  # pjt_faster_rcnn_250126_OCR_CHAR3_0p7 vgg
#         # filepath_model_char=os.path.join(dirpath, r'trained/char/fasterrcnn_01270249_0.9445500180625707'),  # pjt_faster_rcnn_250126_OCR_CHAR3_0p9 vgg

#         #-------------------------------------------------------
#         n_test_pre_nms=12000, n_test_post_nms=2000,
#         label_names_char = list('''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+-={}|[]\:";'<>?,./'''), # full except: `
#     )

#     dirpath = os.path.join(r'C:\Users\bomso\bomsoo1\python\_pytorch\pjt_faster_rcnn', 'test_data')
#     # dirpath = os.path.join(r'C:\Users\bomso\bomsoo1\python\_pytorch\pjt_faster_rcnn', 'test_images_BOC')
#     # for i, filename in enumerate(os.listdir(dirpath)):
#     for i, filename in enumerate([
#         # 'I-20_Sample-UA-e1651007730153.jpg',
#         'KakaoTalk_20240608_233736903.jpg',
#         # 'pic01.png',
#         # 'pic05.png',
#         # 'pic06.png',
#         # 'pic09.png',
#         # 'pic11.png',
#         # 'opening_skinner.png',
#         # 'driver_license_00.png', # test_images_BOC
#         ]):

#         print(f"[{i+1}] Filename = {filename}, time = {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         img = read_image(os.path.join(dirpath, filename)) # load image

#         out = ocr_model.predict(img, fix_img_H=100, debug=False)


