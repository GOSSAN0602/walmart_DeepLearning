import numpy as np
import pandas as pd
import argparse
import datetime
import json

import matplotlib.pyplot as plt
import os
import gc
import sys
sys.path.append("./")
from libs.model import *
from libs.trainer import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, normalize
from sklearn.metrics import mean_squared_error


# set params
parser = argparse.ArgumentParser(description='Walmart NN')
parser.add_argument('--debug', type=bool, default=False, help='Length Train Data')
parser.add_argument('--INPUT_DIR', type=str, default='../../input/lv9/', help='Dataset dir')
parser.add_argument('--state', type=str, default='WI', help='data of state')
parser.add_argument('--use_days', type=int, default=365*3, help='Length Train Data')
parser.add_argument('--n_epoch', type=int, default=250)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--optimizer', type=str, default='Adam', help='choose from Adam, RAdam, SGD')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='param of adam')
parser.add_argument('--beta2', type=float, default=0.7, help='param of adam')
args = parser.parse_args()

# log
log_name = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
log_dir = f'./log/{log_name}'
os.mkdir(log_dir)
with open(f"{log_dir}/params.json", mode="w") as f:
    json.dump(args.__dict__, f, indent=4)
#os.mkdir(f'../model_weight/{log_name}')

# load data
sales_array = np.load(args.INPUT_DIR+'sales_array.npy') # (70,1,1936)
feat_array = np.load(args.INPUT_DIR+'feat_array.npy') # (70,7,1964)

# make numeric input
sales_train_x = sales_array[:, 0, -1 * (28 + args.use_days) : -28*2]
sales_train_t = sales_array[:, 0, -28 * 2:-28]
sales_valid_x = sales_array[:, 0, -1 * args.use_days : -28]
sales_valid_t = sales_array[:, 0, -28:]
sales_test_x = sales_array[:, 0, -1 * (args.use_days - 28) :]

feat_train_x = feat_array[:, :, -1 * (28 * 2 + args.use_days) : -28 * 2]
feat_valid_x = feat_array[:, :, -1 * (28 + args.use_days) : -28]
feat_test_x = feat_array[:, :, -1 * args.use_days : ]

# preprocess for NN
def cut_outrange(input, min=0.05, max=0.95):
    input = np.where(min > input, 0.05, input)
    return np.where(max < input, max, input)

mm = MinMaxScaler(feature_range=(0.05, 0.95))
mm.fit(sales_train_x.T)
# sales transform & padding x last 28days
padding_array = np.zeros([70,28])
sales_train_x_norm_pad = np.concatenate([mm.transform(sales_train_x.T).T, padding_array], axis=1).reshape(70,1,args.use_days)
sales_train_t_norm = mm.transform(sales_train_t.T).T.reshape(70,1,28)
sales_valid_x_norm_pad = np.concatenate([mm.transform(sales_valid_x.T).T, padding_array], axis=1).reshape(70,1,args.use_days)
sales_valid_t_norm = mm.transform(sales_valid_t.T).T.reshape(70,1,28)
sales_test_x_norm_pad = np.concatenate([mm.transform(sales_test_x.T).T, padding_array], axis=1).reshape(70,1,args.use_days)

mm = MinMaxScaler(feature_range=(0.05, 0.95))
# feat transform
feat_train_x_norm = np.zeros(feat_train_x.shape)
feat_valid_x_norm = np.zeros(feat_train_x.shape)
feat_test_x_norm = np.zeros(feat_train_x.shape)
for i in range(70):
    feat_train_x_norm[i] = mm.fit_transform(feat_train_x[i].T).T
    feat_valid_x_norm[i] = mm.transform(feat_valid_x[i].T).T
    feat_test_x_norm[i] = mm.transform(feat_test_x[i].T).T

# merge sells & days
n_dyn_fea = 8
input_train_x = np.concatenate([sales_train_x_norm_pad, feat_train_x_norm],axis=1)
input_train_t = sales_train_t_norm
input_valid_x = np.concatenate([sales_valid_x_norm_pad, feat_valid_x_norm],axis=1)
input_valid_t = sales_valid_t_norm
input_test_x = np.concatenate([sales_test_x_norm_pad, feat_test_x_norm],axis=1)

# numpy -> tensor
tr_x = torch.from_numpy(input_train_x).float()
tr_t = torch.from_numpy(input_train_t).float()
va_x = torch.from_numpy(input_valid_x).float()
va_t = torch.from_numpy(input_valid_t).float()
te_x = torch.from_numpy(input_test_x).float()

# define NN
#my_model = dilated_CNN(args, n_dyn_fea)
my_model = amane_wavenet(args, n_dyn_fea)
dilated_cnn_trainer(args, my_model, tr_x, tr_t, va_x, va_t, te_x, log_dir)
