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
parser.add_argument('--INPUT_DIR', type=str, default='../../input/lv3', help='Dataset dir')
parser.add_argument('--use_days', type=int, default=365*3, help='Length Train Data')
parser.add_argument('--n_epoch', type=int, default=250)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='Adam', help='choose from Adam, RAdam, SGD')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
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
n_dyn_fea = 10
store_list = ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3', 'WI_1','WI_2','WI_3']
input_array = np.zeros([len(store_list), 5, 1934])
price_array = np.zeros([len(store_list), 2, 1966])
for i, store in enumerate(store_list):
    data = pd.read_csv(f'{args.INPUT_DIR}/sales_lv3_{store}.csv')
    input_array[i,0] = data.drop(['1061','1426','1791'], axis=1).iloc[1,-1934:].values
    input_array[i,1:3] = np.load(f'{args.INPUT_DIR}/sales_diff_{store}.npy')
    price_data = pd.read_csv(f'{args.INPUT_DIR}/price_fea_{store}.csv')
    price_data = price_data.drop('index',axis=1)
    price_array[i] = price_data.drop(0,axis=0).drop(price_data.columns[[1061,1426,1791]], axis=1)
    sales_minmax = pd.read_csv(f'{args.INPUT_DIR}/min_max_{store}.csv')
    sales_minmax = sales_minmax.drop('index',axis=1)
    sales_minmax = sales_minmax.drop(0,axis=0)
    input_array[i,3:5] = sales_minmax.drop(['1061','1426','1791'], axis=1).iloc[:,-1934:].values

# make days array
days_data = pd.read_csv(f'{args.INPUT_DIR}/cal_feat_v2.csv')
for col in days_data.columns:
    days_data.loc[:,col] /= days_data.max()[col]
days_data = days_data.T
days_data = days_data.drop(days_data.columns[[1061,1426,1791]],axis=1)

days_array = np.zeros([len(store_list), 5, 1966])
for i in range(4):
    tmp_days_data = days_data.T.loc[:,['wday','sports','ramadan','day_type','snap_CA']]
    tmp_days_data = tmp_days_data.T
    days_array[i] = tmp_days_data.values
for i in range(3):
    tmp_days_data = days_data.T.loc[:,['wday','sports','ramadan','day_type','snap_TX']]
    tmp_days_data = tmp_days_data.T
    days_array[i+4] = tmp_days_data.values
for i in range(3):
    tmp_days_data = days_data.T.loc[:,['wday','sports','ramadan','day_type','snap_WI']]
    tmp_days_data = tmp_days_data.T
    days_array[i+7] = tmp_days_data.values

# make numeric input
train_x = input_array[:, :, -1 * (28 + args.use_days) : -28*2].copy()
train_t = input_array[:,  0, -28 * 2:-28].reshape(-1,1,28).copy()
valid_x = input_array[:, :, -1 * args.use_days : -28].copy()
valid_t = input_array[:, 0, -28:].reshape(-1,1,28).copy()
test_x = input_array[:, :, -1 * (args.use_days - 28) :].copy()
train_days_x = days_array[:, :, -1 * (28 * 2 + args.use_days) : -28 * 2].copy()
valid_days_x = days_array[:, :, -1 * (28 + args.use_days) : -28].copy()
test_days_x = days_array[:, :, -1 * args.use_days : ].copy()

del input_array, days_array
gc.collect()

# preprocess for NN
# def cut_outrange(input, min=0.05, max=0.95):
#     input = np.where(min > input, 0.05, input)
#     return np.where(max < input, max, input)

mm_sales_list = []
mm_others_list = []
for i in range(len(store_list)):
    mm_sales = MinMaxScaler(feature_range=(0.05, 0.95))
    mm_sales.fit(train_x[i,0].reshape(1,-1).T)
    mm_sales_list.append(mm_sales)
    # transform & padding last 28days
    train_x[i,0] = mm_sales.transform(train_x[i,0].reshape(1,-1).T).T
    train_t[i,0] = mm_sales.transform(train_t[i,0].reshape(1,-1).T).T.reshape(-1,)
    valid_x[i,0] = mm_sales.transform(valid_x[i,0].reshape(1,-1).T).T.reshape(-1,)
    valid_t[i,0] = mm_sales.transform(valid_t[i,0].reshape(1,-1).T).T.reshape(-1,)
    test_x[i,0] = mm_sales.transform(test_x[i,0].reshape(1,-1).T).T.reshape(-1,)

    mm_others = MinMaxScaler(feature_range=(0.05, 0.95))
    mm_others.fit(train_x[i,1:].T)
    mm_others_list.append(mm_others)
    # transform & padding last 28days
    train_x[i,1:] = mm_others.transform(train_x[i,1:].T).T
    valid_x[i,1:] = mm_others.transform(valid_x[i,1:].T).T
    test_x[i,1:] = mm_others.transform(test_x[i,1:].T).T

#padding last 28days
padding_array = np.zeros([10,5,28])
train_x = np.concatenate([train_x, padding_array],axis=2)
valid_x = np.concatenate([valid_x, padding_array],axis=2)
test_x = np.concatenate([test_x, padding_array],axis=2)

input_train_x = np.concatenate([train_x, train_days_x],axis=1)
input_valid_x = np.concatenate([valid_x, valid_days_x],axis=1)
input_test_x = np.concatenate([test_x, test_days_x],axis=1)

# # check input data
# for i in range(10):
#     plt.plot(input_test_x[i,0])
# plt.show()

# numpy -> tensor
tr_x = torch.from_numpy(input_train_x).float()
tr_t = torch.from_numpy(train_t).float()
va_x = torch.from_numpy(input_valid_x).float()
va_t = torch.from_numpy(valid_t).float()
te_x = torch.from_numpy(input_test_x).float()

# define NN
#my_model = dilated_CNN(args, n_dyn_fea)
my_model = amane_wavenet(args, n_dyn_fea)
dilated_cnn_trainer(args, my_model, tr_x, tr_t, va_x, va_t, te_x, log_dir)
