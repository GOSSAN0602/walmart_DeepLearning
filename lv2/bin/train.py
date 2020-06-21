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
parser.add_argument('--state', type=str, default='WI', help='data of state')
parser.add_argument('--use_days', type=int, default=365, help='Length Train Data')
parser.add_argument('--n_epoch', type=int, default=300)
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
days_data = pd.read_csv(f'{args.INPUT_DIR}/cal_feat_v2.csv')

days_data = days_data.loc[:,['wday','sports','ramadan','day_type','snap_'+args.state]]
for col in days_data.columns:
    days_data.loc[:,col] /= days_data.max()[col]
days_data = days_data.T
# remove Christmas
data = data.drop(['d_1062','d_1427','d_1792'],axis=1)
days_data = days_data.drop(days_data.columns[[1061,1426,1791]],axis=1)

# make numeric input
train_x = np.array(data.iloc[:, -1 * (28*2 + args.use_days) : -28*2]).reshape(1,-1)
#train_x = np.array(data.iloc[:, -1 * (28 + args.use_days) : -28]).reshape(3,-1)
train_t = np.array(data.iloc[:,  -28 * 2:-28]).reshape(1,-1)
valid_x = np.array(data.iloc[:, -1 * (28 + args.use_days) : -28]).reshape(1,-1)
valid_t = np.array(data.iloc[:,  -28:]).reshape(1,-1)

train_days_x = np.array(days_data.iloc[:, -1 * (28 * 3 + args.use_days) : -28 * 3])
# train_days_x = np.array(days_data.iloc[:, -1 * (28 * 2 + args.use_days) : -28 * 2])
valid_days_x = np.array(days_data.iloc[:, -1 * (28 * 2 + args.use_days) : -28 * 2])

# preprocess for NN
def cut_outrange(input, min=0.05, max=0.95):
    input = np.where(min > input, 0.05, input)
    return np.where(max < input, max, input)

mm = MinMaxScaler(feature_range=(0.05, 0.95))
mm.fit(train_x.T)
train_x = mm.transform(train_x.T).T
train_t = cut_outrange(mm.transform(train_t.T).T)
valid_x = cut_outrange(mm.transform(valid_x.T).T)
valid_t = cut_outrange(mm.transform(valid_t.T).T)

# merge sells & days
n_dyn_fea = 6
input_train_x = np.concatenate([train_x, train_days_x]).reshape(1,n_dyn_fea,args.use_days)
input_train_t = train_t.reshape(1,28)
input_valid_x = np.concatenate([valid_x, valid_days_x]).reshape(1,n_dyn_fea,args.use_days)
input_valid_t = valid_t.reshape(1,28)


# numpy -> tensor
tr_x = Variable(torch.from_numpy(input_train_x).float(), requires_grad=True)
tr_t = Variable(torch.from_numpy(input_train_t).float(), requires_grad=False)
va_x = Variable(torch.from_numpy(input_valid_x).float(), requires_grad=True)
va_t = Variable(torch.from_numpy(input_valid_t).float(), requires_grad=False)

# define NN
my_model = amane_wavenet(args, n_dyn_fea)
dilated_cnn_trainer(args, my_model, tr_x, tr_t, va_x, va_t, log_dir, mm)
