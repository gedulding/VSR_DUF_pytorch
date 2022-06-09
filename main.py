import os
import torch
import time
import logging
import warnings
import argparse
import glob
import train
import test
import numpy as np
import scipy.ndimage.filters as fi
from torch.utils.data import TensorDataset, DataLoader


from torchvision.transforms import transforms
from utils import LoadImage, loadyaml, load_model

warnings.filterwarnings("ignore")

# Parameter(default)

R = 4                                                       # Upscaling factor
pixel_x = 144
pixel_y = 180
net_Depth = 16                                              # network depth : 16(default) 28 52

pt_dir = './model'

data_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(),                         #Flip picture randomly
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def main():
    # input Parameter
    parser = argparse.ArgumentParser()

    parser.add_argument('--R', type=int, help='Upscaling factor: One of 2, 3, 4')
    parser.add_argument('--L',  type=int, help='Network depth: One of 16, 28, 52')
    parser.add_argument('--resume' ,default = False,action='store_true',help='If set resumes training from provided checkpoint. (default: False)')
    parser.add_argument('--path_to_checkpoint',type=str,default='./model/model_checkpoint.ckpt',help='Path to checkpoint to resume training. (default: "")')
    args = parser.parse_args()

    # ---------------- train model -------------------------------
    # load train data dataset
    yamfile = './config/config.yaml'
    config = loadyaml(yamfile)
    train_data_path = config['data']['train_data']
    train_labels_path = config['data']['train_labels']
    test_data_path = config['data']['test_data']
    test_labels_path = config['data']['test_labels']
    num_workers = int(config['data']['num_workers'])
    train_mini_batchSize = int(config['train']['batch_size'])
    test_mini_batchSize = int(config['test']['batch_size'])
    T = int(config['train']['T_in'])                                                # N = 3(default)

    train_path_to_data = train_data_path
    dir_frames = glob.glob(train_path_to_data + '/*.png')
    dir_frames.sort()
    frames = []
    for f in dir_frames:
        img = LoadImage(f)
        img = fi.gaussian_filter(img , 1.6)                                         # Gaussian smoothing
        frames.append(img)
    frames = np.asarray(frames)                                                     # N * C * H * W

    frames_padded = np.lib.pad(frames, pad_width=((T // 2, T // 2), (0, 0), (0, 0), (0, 0)),                   # N+6 * C * H * W
                               mode='constant')
    if R == 2:
        frames_padded = np.lib.pad(frames_padded, pad_width=((0, 0), (0, 0),(2 * R, 2 * R), (2 * R, 2 * R)),
                                   mode='reflect')
    elif R == 3:
        H_h, H_w = frames.shape[1:3]
        pad_h = 3 - (H_h % 3)
        pad_w = 3 - (H_w % 3)
        frames_padded = np.lib.pad(frames_padded,
                                   pad_width=((0, 0), (0, 0),(2 * R, 2 * R + pad_h), (2 * R, 2 * R + pad_w)),
                                   mode='reflect')
    elif R == 4:
        frames_padded = np.lib.pad(frames_padded, pad_width=((0, 0), (0, 0) ,(2 * R, 2 * R), (2 * R, 2 * R)),
                                   mode='reflect')
    in_L = []
    for i in range(frames.shape[0]):
        in_L.append(frames_padded[i:i+T])                   # select T frames
    in_L = np.asarray(in_L)

    in_L = in_L.reshape([in_L.shape[0] , in_L.shape[2] ,in_L.shape[1] ,in_L.shape[3] ,in_L.shape[4]])             #N * C * 7 * H * W

    tensor_x = torch.from_numpy(in_L)
    train_data_dataset = TensorDataset(tensor_x)
    train_data_loader = DataLoader(train_data_dataset, batch_size=train_mini_batchSize, shuffle=False,num_workers=num_workers)       #B * C * 7 * H * W
    # load train labels dataset
    train_path_to_labels = train_labels_path
    dir_labels_frames = glob.glob(train_path_to_data + '/*.png')
    dir_labels_frames.sort()
    labels_frames = []
    for f in dir_labels_frames:
        labels_frames.append(LoadImage(f))
    labels_frames = np.asarray(labels_frames)
    tensor_y = torch.from_numpy(labels_frames)
    train_labels_dataset = TensorDataset(tensor_y)
    train_labels_loader = DataLoader(train_labels_dataset, batch_size=train_mini_batchSize, shuffle=False,num_workers=num_workers)
    train.train(args , pt_dir , train_data_loader , train_labels_loader)

    # ---------------- test eval -------------------------------
    test_dataset = test_data_path
    dir_test_frames = glob.glob(test_dataset + '/*.png')
    dir_test_frames.sort()
    test_data_frames = []
    for f in dir_test_frames:
        test_data_frames.append(LoadImage(f))
    test_data_frames = np.asarray(test_data_frames)
    test_data_frames_padded = np.lib.pad(test_data_frames, pad_width=((T // 2, T // 2), (0, 0), (0, 0), (0, 0)),
                               mode='constant')                                                                 # 补帧，补前面和后面的帧，要输入7个帧，不够
    if R == 2:
        test_data_frames_padded = np.lib.pad(test_data_frames_padded, pad_width=((0, 0), (0, 0), (2 * R, 2 * R), (2 * R, 2 * R)),
                                   mode='reflect')
    elif R == 3:
        H_h, H_w = test_data_frames.shape[1:3]
        pad_h = 3 - (H_h % 3)
        pad_w = 3 - (H_w % 3)
        test_data_frames_padded = np.lib.pad(test_data_frames_padded,
                                   pad_width=((0, 0), (0, 0), (2 * R, 2 * R + pad_h), (2 * R, 2 * R + pad_w)),
                                   mode='reflect')
    elif R == 4:
        test_data_frames_padded = np.lib.pad(test_data_frames_padded, pad_width=((0, 0), (0, 0), (2 * R, 2 * R), (2 * R, 2 * R)),
                                   mode='reflect')  # 做对称填充
    test_in_L = []
    for i in range(frames.test_data_frames[0]):
        test_in_L.append(test_data_frames_padded[i:i + T])                                                      # select T frames
    test_in_L = np.asarray(test_in_L)
    test_in_L = test_in_L.reshape([test_in_L.shape[0], test_in_L.shape[2], test_in_L.shape[1], test_in_L.shape[3], test_in_L.shape[4]])

    test_tensor_x = torch.from_numpy(test_in_L)
    test_data_dataset = TensorDataset(test_tensor_x)
    test_data_loader = DataLoader(test_data_dataset, batch_size=test_mini_batchSize, shuffle=False,num_workers=num_workers)

    # load test labels dataset
    test_path_to_labels = test_labels_path
    dir_test_labels_frames = glob.glob(train_path_to_data + '/*.png')
    dir_test_labels_frames.sort()
    test_labels_frames = []
    for f in dir_test_labels_frames:
        test_labels_frames.append(LoadImage(f))
    test_labels_frames = np.asarray(test_labels_frames)
    test_tensor_y = torch.from_numpy(test_labels_frames)
    test_labels_dataset = TensorDataset(test_tensor_y)
    test_labels_loader = DataLoader(test_labels_dataset, batch_size=test_mini_batchSize, shuffle=False,num_workers=num_workers)

    # load model
    model_filepath = config['test']['model_filepath']
    model = load_model(model_filepath)
    test.test(model, test_data_loader, test_labels_loader)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
