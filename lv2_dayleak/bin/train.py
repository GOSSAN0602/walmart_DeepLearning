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
parser.add_argument('--INPUT_DIR', type=str, default='../../input/lv2', help='Dataset dir')
parser.add_argument('--state', type=str, default='TX', help='data of state')
parser.add_argument('--use_days', type=int, default=365*3, help='Length Train Data')
parser.add_argument('--n_epoch', type=int, default=250)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
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
data = pd.read_csv(f'{args.INPUT_DIR}/sales_lv2.csv')
data = data[data['index']==args.state]
sales_diff = np.load(f'{args.INPUT_DIR}/sales_diff_{args.state}.npy')
days_data = pd.read_csv(f'{args.INPUT_DIR}/cal_feat_v2.csv')
price_data = pd.read_csv(f'{args.INPUT_DIR}//price_fea_{args.state}.csv')
price_data = price_data.drop('index',axis=1)
price_data = price_data.drop(0,axis=0)
sales_minmax = pd.read_csv(f'{args.INPUT_DIR}/min_max_{args.state}.csv')
sales_minmax = sales_minmax.drop('index',axis=1)
sales_minmax = sales_minmax.drop(0,axis=0)

days_data['snap']=days_data.iloc[:,3:6].sum(axis=1)
days_data = days_data.loc[:,['wday','sports','ramadan','day_type','snap']]
for col in days_data.columns:
    days_data.loc[:,col] /= days_data.max()[col]
days_data = days_data.T

# remove Christmas
data = data.drop(['d_1062','d_1427','d_1792'],axis=1)
days_data = days_data.drop(days_data.columns[[1061,1426,1791]],axis=1)
price_data = price_data.drop(price_data.columns[[1061,1426,1791]], axis=1)
sales_minmax = sales_minmax.drop(sales_minmax.columns[[1061,1426,1791]], axis=1)

# make numeric input
train_x = np.array(data.iloc[:, -1 * (28 + args.use_days) : -28*2]).sum(axis=0).reshape(1,-1)
train_t = np.array(data.iloc[:,  -28 * 2:-28]).sum(axis=0).reshape(1,-1)
valid_x = np.array(data.iloc[:, -1 * args.use_days : -28]).sum(axis=0).reshape(1,-1)
valid_t = np.array(data.iloc[:,  -28:]).sum(axis=0).reshape(1,-1)
test_x = np.array(data.iloc[:, -1 * (args.use_days - 28) :]).sum(axis=0).reshape(1,-1)

train_sales_diff = sales_diff[:,-1 * (28 + args.use_days) : -28*2]
valid_sales_diff = sales_diff[:,-1 * args.use_days : -28]
test_sales_diff = sales_diff[:, -1 * (args.use_days - 28) :]

train_days_x = np.array(days_data.iloc[:, -1 * (28 * 2 + args.use_days) : -28 * 2])
valid_days_x = np.array(days_data.iloc[:, -1 * (28 + args.use_days) : -28])
test_days_x = np.array(days_data.iloc[:, -1 * args.use_days : ])

train_price_x = np.array(price_data.iloc[:, -1 * (28 * 2 + args.use_days) : -28 * 2])
valid_price_x = np.array(price_data.iloc[:, -1 * (28 + args.use_days) : -28])
test_price_x = np.array(price_data.iloc[:, -1 * args.use_days : ])

train_sales_minmax = np.array(sales_minmax.iloc[:,-1 * (28 + args.use_days) : -28*2])
valid_sales_minmax = np.array(sales_minmax.iloc[:,-1 * args.use_days : -28])
test_sales_minmax = np.array(sales_minmax.iloc[:, -1 * (args.use_days - 28) :])

# preprocess for NN
def cut_outrange(input, min=0.05, max=0.95):
    input = np.where(min > input, 0.05, input)
    return np.where(max < input, max, input)

mm = MinMaxScaler(feature_range=(0.05, 0.95))
mm.fit(train_x.T)
# transform & padding last 28days
padding_array = np.zeros([1,28])
train_x = np.concatenate([mm.transform(train_x.T).T, padding_array], axis=1)
train_t = mm.transform(train_t.T).T
valid_x = np.concatenate([mm.transform(valid_x.T).T, padding_array], axis=1)
valid_t = mm.transform(valid_t.T).T
test_x = np.concatenate([mm.transform(test_x.T).T, padding_array], axis=1)

padding_array = np.zeros([2,28])
train_sales_minmax = np.concatenate([mm.transform(train_sales_minmax.T).T, padding_array], axis=1)
valid_sales_minmax = np.concatenate([mm.transform(valid_sales_minmax.T).T, padding_array], axis=1)
test_sales_minmax = np.concatenate([mm.transform(test_sales_minmax.T).T, padding_array], axis=1)

# transform & padding last 28days
padding_array = np.zeros([2,28])
mm = MinMaxScaler(feature_range=(0.05, 0.95))
mm.fit(train_sales_diff.T)
train_sales_diff = np.concatenate([mm.transform(train_sales_diff.T).T, padding_array], axis=1)
valid_sales_diff = np.concatenate([mm.transform(valid_sales_diff.T).T, padding_array], axis=1)
test_sales_diff = np.concatenate([mm.transform(test_sales_diff.T).T, padding_array], axis=1)

mm = MinMaxScaler(feature_range=(0.05, 0.95))
mm.fit(train_price_x.T)
train_price_x = mm.transform(train_price_x.T).T
valid_price_x = mm.transform(valid_price_x.T).T
test_price_x = mm.transform(test_price_x.T).T

# merge sells & days
n_dyn_fea = 12
input_train_x = np.concatenate([train_x, train_sales_minmax, train_sales_diff, train_days_x, train_price_x]).reshape(1,n_dyn_fea,args.use_days)
input_train_t = train_t.reshape(1,28)
input_valid_x = np.concatenate([valid_x, valid_sales_minmax, valid_sales_diff, valid_days_x, valid_price_x]).reshape(1,n_dyn_fea,args.use_days)
input_valid_t = valid_t.reshape(1,28)
input_test_x = np.concatenate([test_x, test_sales_minmax, test_sales_diff, test_days_x, test_price_x]).reshape(1, n_dyn_fea, args.use_days)

# n_dyn_fea = 6
# input_train_x = np.concatenate([train_x, train_days_x]).reshape(1,n_dyn_fea,args.use_days)
# input_train_t = train_t.reshape(1,28)
# input_valid_x = np.concatenate([valid_x, valid_days_x]).reshape(1,n_dyn_fea,args.use_days)
# input_valid_t = valid_t.reshape(1,28)
# input_test_x = np.concatenate([test_x, test_days_x]).reshape(1, n_dyn_fea, args.use_days)


# numpy -> tensor
tr_x = torch.from_numpy(input_train_x).float()
tr_t = torch.from_numpy(input_train_t).float()
va_x = torch.from_numpy(input_valid_x).float()
va_t = torch.from_numpy(input_valid_t).float()
te_x = torch.from_numpy(input_test_x).float()

# define NN
#my_model = dilated_CNN(args, n_dyn_fea)
my_model = amane_wavenet(args, n_dyn_fea)
dilated_cnn_trainer(args, my_model, tr_x, tr_t, va_x, va_t, te_x, log_dir, mm)
