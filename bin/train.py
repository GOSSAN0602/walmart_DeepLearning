import numpy as np
import pandas as pd
import argparse

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



# set params
parser = argparse.ArgumentParser(description='Walmart NN')
parser.add_argument('--debug', type=bool, default=True, help='Length Train Data')
parser.add_argument('--use_days', type=int, default=365, help='Length Train Data')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='Adam', help='choose from Adam, RAdam, SGD')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='param of adam')
parser.add_argument('--beta2', type=float, default=0.999, help='param of adam')
args = parser.parse_args()

# load data
data = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
if args.debug:
    print('*******DEBUG*********')
    data = data[(data['store_id']=='CA_1') & (data['dept_id']=='HOBBIES_1')]
train_x = np.array(data.iloc[:, -1 * (28 * 2 + args.use_days) : -28 * 2])
train_t = np.array(data.iloc[:,  -28 * 2:-28])
valid_x = np.array(data.iloc[:, -1 * (28 + args.use_days) : -28])
valid_t = np.array(data.iloc[:,  -28:])

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

# reshape to (Batch, Dim, Length)
train_x = train_x.reshape(-1,1,args.use_days)
train_t = train_t.reshape(-1,1,28)
valid_x = valid_x.reshape(-1,1,args.use_days)
valid_t = valid_t.reshape(-1,1,28)

# numpy -> tensor
tr_x = Variable(torch.from_numpy(train_x).float(), requires_grad=True)
tr_t = Variable(torch.from_numpy(train_t).float(), requires_grad=False)
va_x = Variable(torch.from_numpy(valid_x).float(), requires_grad=True)
va_t = Variable(torch.from_numpy(valid_t).float(), requires_grad=False)

# define NN
my_model = simple_Net(args)
rnn_trainer(args, my_model, tr_x, tr_t, va_x, va_t)