import numpy as np
import pandas as pd
import argparse
import datetime

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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error


# set params
parser = argparse.ArgumentParser(description='Walmart NN')
parser.add_argument('--debug', type=bool, default=True, help='Length Train Data')
parser.add_argument('--INPUT_DIR', type=str, default='../input/m5-forecasting-accuracy', help='Dataset dir')
parser.add_argument('--store_id', type=str, default='CA_1', help='data of store_id')
parser.add_argument('--use_days', type=int, default=365, help='Length Train Data')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam', help='choose from Adam, RAdam, SGD')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='param of adam')
parser.add_argument('--beta2', type=float, default=0.999, help='param of adam')
args = parser.parse_args()

# log
log_name = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
log_dir = f'./log/{log_name}'
os.mkdir(log_dir)
#os.mkdir(f'../model_weight/{log_name}')

# load data
iter_csv = pd.read_csv(f'{args.INPUT_DIR}/sales_train_validation.csv', iterator=True, chunksize=1000)
data = pd.concat([chunk[chunk['store_id'] == args.store_id] for chunk in iter_csv])
if args.debug:
    print('*******DEBUG*********')
    data = data.sample(n=args.batch_size*2, random_state=0)

# make categorical input
c_list = ["dept_id","cat_id"]
MAX_CAT_ID = len(data["cat_id"].unique())
MAX_DEPT_ID = len(data["dept_id"].unique())
cat_array = np.zeros([data.shape[0],len(c_list)])
for i, c in enumerate(c_list):
    le = LabelEncoder()
    cat_array[:,i] = le.fit_transform(data[c])

# make numeric input
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
cat_input = Variable(torch.from_numpy(cat_array).float(), requires_grad=True).long()

# define NN
my_model = dilated_CNN(args, 1, MAX_CAT_ID, MAX_DEPT_ID)
import pdb;pdb.set_trace()
dilated_cnn_trainer(args, my_model, tr_x, cat_input, tr_t, va_x, va_t, log_dir)