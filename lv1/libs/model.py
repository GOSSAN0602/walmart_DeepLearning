import argparse

import numpy as np
from math import ceil, log

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Conv1d, Linear, Embedding, Flatten, Dropout
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

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)

class dilated_CNN(nn.Module):
    def __init__(self, args, n_dyn_fea, MAX_CAT_ID, MAX_DEPT_ID):
        super(dilated_CNN, self).__init__(
        )
        # params
        seq_len = args.use_days
        self.n_dyn_fea = n_dyn_fea
        self.n_dilated_layers = 3
        kernel_size = 2
        n_filters = 3
        max_cat_id = [MAX_DEPT_ID, MAX_CAT_ID]
        n_outputs = 28
        dropout_rate = 0.1

        # layers for categorical input
        self.lambda0 = LambdaLayer(lambda x : x[:, 0])
        self.lambda1 = LambdaLayer(lambda x : x[:, 1])
        self.embedding0 = Embedding(max_cat_id[0]+1, ceil(log(max_cat_id[0]+1)))
        self.embedding1 = Embedding(max_cat_id[1]+1, ceil(log(max_cat_id[1]+1)))
        self.flatten0 = Flatten()
        self.flatten1 = Flatten()

        # Dilated convolutional layers
        self.conv1d = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=kernel_size, dilation=1)
        self.conv1d_dilated0 = CausalConv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=2)
        self.conv1d_dilated1 = CausalConv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=2**2)
        self.conv1d_dilated2 = CausalConv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=2**3)

        # conv output layers
        self.conv1d_out = CausalConv1d(in_channels=n_filters*2 , out_channels=8, kernel_size=1)
        self.dropout_out = Dropout(dropout_rate)
        self.flatten_out = Flatten()

        # layers for concatenating with cat and num features
        self.dense_concat0 = Linear(in_features=1669, out_features=56)
        self.dense_concat1 = Linear(in_features=56, out_features=n_outputs)
    
    def __call__(self, seq_in, cat_fea_in, t, criterion):
        pred = self.forward(seq_in, cat_fea_in)
        loss = criterion(pred, t)
        return loss, pred
    
    def forward(self, seq_in, cat_fea_in):
        # Categorical input
        cat_flatten = []
        cat_fea = self.lambda0(cat_fea_in)
        cat_fea_embed = self.embedding0(cat_fea)
        cat_flatten.append(self.flatten0(cat_fea_embed))
        cat_fea = self.lambda1(cat_fea_in)
        cat_fea_embed = self.embedding1(cat_fea)
        cat_flatten.append(self.flatten1(cat_fea_embed))

        # conv layers
        h0 = F.relu(self.conv1d(seq_in))
        h1 = F.relu(self.conv1d_dilated0(h0))
        h2 = F.relu(self.conv1d_dilated1(h1))
        h3 = F.relu(self.conv1d_dilated2(h2))

        # Skip connections
        c = torch.cat((h0, h3), 1)

        # out put of conv layers
        conv_out = F.relu(self.conv1d_out(c))
        conv_out = F.relu(self.dropout_out(conv_out))
        conv_out = F.relu(self.flatten_out(conv_out))

        # Concatenate with categorical features
        x = torch.cat((conv_out, cat_flatten[0], cat_flatten[1]), 1)
        x = F.relu(self.dense_concat0(x))
        output = self.dense_concat1(x)

        return output