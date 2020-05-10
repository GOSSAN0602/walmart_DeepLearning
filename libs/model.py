import argparse

import numpy as np
from math import ceil, log

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Conv1d, Dense, Embedding, Flatten, Dropout
import torch.nn.functional as F


class simple_Net(nn.Module):
    def __init__(self, args):
        super(simple_Net, self).__init__()
        self.layer1 = nn.LSTM(1,1,1,True,False,0,False)
    
    def __call__(self, x, t, criterion):
        h0 = torch.randn(1, x.shape[0], 1)
        c0 = torch.randn(1, x.shape[0], 1)
        steps = t.shape[2]
        (hn, cn) = self.open_loop(x,h0,c0)
        loss, pred_seq = self.closed_loop(x, hn, cn, t, steps, criterion)
        return loss, pred_seq

    def open_loop(self, x,h0,c0):
        _, (hn, cn) = self.layer1(x[:,:,0].reshape(1,-1,1), (h0, c0))
        for i in range(x.shape[2]-1):
            _, (hn, cn) = self.layer1(x[:,:,i+1].reshape(1,-1,1), (hn, cn))
        return (hn, cn)
    
    def closed_loop(self, x, hn, cn, t, steps, criterion):
        pred_seq = np.zeros([t.shape[0], steps])
        loss = 0.0
        _x, (hn, cn) = self.layer1(x[:,:,-1].reshape(1,-1,1), (hn, cn))
        loss += criterion(_x, t[:,:,0:1].reshape(1,-1,1))
        pred_seq[:,0:1] = _x.detach().numpy()
        for i in range(steps-1):
            _x, (hn, cn) = self.layer1(_x, (hn, cn))
            loss += criterion(_x, t[:,:,(i+1):(i+2)].reshape(1,-1,1))
            pred_seq[:,(i+1):(i+2)] = _x.detach().numpy()
        return loss, pred_seq

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class dilated_CNN(nn.Module):
    def __init__(self, args, n_dyn_fea, MAX_CAT_ID, MAX_DEPT_ID):
        super(dilated_CNN, self).__init__(
        )
        # params
        self.seq_len = args.use_days
        self.n_dyn_fea = 
        self.n_dilated_layers = 3
        self.kernel_size = 2
        self.n_filters = 3
        max_cat_id = [MAX_CAT_ID, MAX_DEPT_ID]
        n_outputs = 28

        # layers for categorical input
        self.lambda0 = LambdaLayer(lambda x : x[:, 0, None])
        self.lambda1 = LambdaLayer(lambda x : x[:, 1, None])
        self.embedding0 = Embedding(max_cat_id[0]+1, ceil(log(max_cat_id[0]+1)))
        self.embedding1 = Embedding(max_cat_id[1]+1, ceil(log(max_cat_id[1]+1)))
        self.flatten0 = Flatten()
        self.flatten1 = Flatten()

        # Dilated convolutional layers
        self.conv1d = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=1, padding="causal", activation="relu")
        self.conv1d_dilated0 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=2, padding="causal", activation="relu")
        self.conv1d_dilated1 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=2**2, padding="causal", activation="relu")
        self.conv1d_dilated2 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=2**3, padding="causal", activation="relu")

        # conv output layers
        self.conv1d_out = Conv1D(8, 1, activation="relu")
        self.dropout_out = Dropout(dropout_rate)
        self.flatten_out = Flatten()

        # layers for concatenating with cat and num features
        self.dense_concat0 = Dense(16, activation="relu")
        self.dense_concat1 = Dense(n_outputs)
    
    def forward(seq_in, cat_fea_in):
        # Categorical input
        cat_flatten = []
        cat_fea = self.lambda0(cat_fea_in)
        cat_fea_embed = self.embedding0(cat_fea)
        cat_flatten.append(self.flatten0(cat_fea_embed))
        cat_fea = self.lambda1(cat_fea_in)
        cat_fea_embed = self.embedding1(cat_fea)
        cat_flatten.append(self.flatten1(cat_fea_embed))

        # conv layers
        h0 = self.conv1d(seq_in)
        h1 = self.conv1d_dilated0(h0)
        h2 = self.conv1d_dilated1(h1)
        h3 = self.conv1d_dilated2(h2)

        # Skip connections
        c = torch.cat((h0, h3), 1)

        # out put of conv layers
        conv_out = self.conv1d_out(c)
        conv_out = self.dropout_out(conv_out)
        conv_out = self.flatten_out(conv_out)

        # Concatenate with categorical features
        x = torch.cat(([conv_out], cat_flatten), 1)
        x = self.dense_concat0(x)
        output = self.dense_concat1(x)

        return output