import numpy as np
import shutil
from PIL import Image
import tensorflow as tf
import torch
import scipy.ndimage.filters as fi
import torch.nn.functional as F
import os
import yaml

from net import DUFNET_16L


def LoadImage(path, color_mode='RGB', channel_mean=None, modcrop=[0,0,0,0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.

    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    img = Image.open(path)

    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:,:,0:1]

    ## To 0-1
    x *= 1.0/255.0

    if channel_mean:
        x[:,:,0] -= channel_mean[0]
        x[:,:,1] -= channel_mean[1]
        x[:,:,2] -= channel_mean[2]

    if modcrop[0]*modcrop[1]*modcrop[2]*modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x.reshape(3,img.size[0],img.size[1])

#Huber Loss
def HuberLoss(y_true, y_pred, delta, axis=None):
    abs_error = torch.abs(y_pred - y_true)
    quadratic = torch.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear

    return torch.mean(losses, axis=axis)

# down load pic
def DownSample(x, scale=4):
    kernlen = 13
    ds_x = x.shape
    x = x.reshape([ds_x[0] * ds_x[2],ds_x[1], ds_x[3], ds_x[4]])
    gass_filer = np.zeros((13,13))
    gass_filer[kernlen // 2, kernlen // 2] = 1
    newgass_filer = fi.gaussian_filter(gass_filer, 1.6)
    newgass_filer = newgass_filer[np.newaxis,np.newaxis,:,:].astype(np.float32)

    # Reflect padding
    W = torch.Tensor(newgass_filer)

    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    depthwise_F = W.repeat([3,3,1,1])

    reflect_x = torch.nn.ReflectionPad2d((pad_top, pad_bottom, pad_left, pad_right))

    x = reflect_x(x)

    c = torch.nn.Conv2d(in_channels=3 , out_channels=3 , kernel_size=filter_width, stride=4, padding=0)
    c.weight.data = depthwise_F
    y = c(x)

    ds_y = y.shape
    y = y.reshape([ds_x[0], 3 , ds_x[2], ds_y[2], ds_y[3]])
    return y

def _rgb2ycbcr(img, maxVal=255):
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr

def to_uint8(x, vmin, vmax):
    x = x.astype('float32')
    x = (x-vmin)/(vmax-vmin)*255 # 0~255
    return np.clip(np.round(x), 0, 255)

#计算PSNR值
def AVG_PSNR(vid_true, vid_pred, vmin=0, vmax=255, t_border=2, sp_border=8, is_T_Y=False, is_P_Y=False):
    '''
        This include RGB2ycbcr and VPSNR computed in Y
    '''
    input_shape = vid_pred.shape
    if is_T_Y:
      Y_true = to_uint8(vid_true, vmin, vmax)
    else:
      Y_true = np.empty(input_shape[:-1])
      for t in range(input_shape[0]):
        Y_true[t] = _rgb2ycbcr(to_uint8(vid_true[t], vmin, vmax), 255)[:,:,0]

    if is_P_Y:
      Y_pred = to_uint8(vid_pred, vmin, vmax)
    else:
      Y_pred = np.empty(input_shape[:-1])
      for t in range(input_shape[0]):
        Y_pred[t] = _rgb2ycbcr(to_uint8(vid_pred[t], vmin, vmax), 255)[:,:,0]

    diff =  Y_true - Y_pred
    diff = diff[t_border: input_shape[0]- t_border, sp_border: input_shape[1]- sp_border, sp_border: input_shape[2]- sp_border]

    psnrs = []
    for t in range(diff.shape[0]):
      rmse = np.sqrt(np.mean(np.power(diff[t],2)))
      psnrs.append(20*np.log10(255./rmse))

    return np.mean(np.asarray(psnrs))


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = DUFNET_16L().cuda()#提前网络结构
    model.load_state_dict(checkpoint['model_state_dict'])#加载网络权重参数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#加载优化器参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

def loadyaml(yamlfile = None):
    if yamlfile is None:
        print("default yaml is not exist")
    try:
        file = open(yamlfile , 'r' , encoding='utf-8')
        cont = file.read()
        config = yaml.load(cont , Loader=yaml.FullLoader)
    except IOError as e:
        print(e)
    finally:
        if file is not None:
            file.close()
    return config
