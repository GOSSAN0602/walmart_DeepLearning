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

class stocastic_dilated_CNN(nn.Module):
    def __init__(self, args, n_dyn_fea):
        super(dilated_CNN, self).__init__(
        )
        # params
        seq_len = args.use_days
        self.n_dyn_fea = n_dyn_fea
        self.n_dilated_layers = 3
        kernel_size = 2
        n_filters = 3
        n_outputs = 28 * 9
        dropout_rate = 0.1

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
        self.dense_concat0 = Linear(in_features=8*args.use_days, out_features=int(8*args.use_days*0.1))
        self.dense_concat1 = Linear(in_features=int(8*args.use_days*0.1), out_features=n_outputs)
    
    def __call__(self, seq_in, t, criterion):
        pred = self.forward(seq_in)
        loss = criterion(pred, t)
        return loss, pred
    
    def forward(self, seq_in):
        # conv layers
        h0 = F.relu(self.conv1d(seq_in))
        h1 = F.relu(self.conv1d_dilated0(h0))
        h2 = F.relu(self.conv1d_dilated1(h1))
        h3 = F.relu(self.conv1d_dilated2(h2))

        # Skip connections
        c = torch.cat((h0, h3), 1)

        # out put of conv layers
        conv_out = F.relu(self.conv1d_out(c))
        # conv_out = F.relu(self.dropout_out(conv_out))
        conv_out = F.relu(self.flatten_out(conv_out))

        # decode
        x = self.dense_concat0(conv_out)
        output = self.dense_concat1(x)

        return output.view(-1,9,28)

class stocastic_kaggler_wavenet(nn.Module):
    def __init__(self, args, n_dyn_fea):
        super(kaggler_wavenet, self).__init__(
        )
        # params
        seq_len = args.use_days
        self.n_dyn_fea = n_dyn_fea
        self.n_dilated_layers = 3
        kernel_size = 2
        n_filters = 3
        n_outputs = 28*9
        dropout_rate = 0.1

        # Dilated convolutional layers
        self.conv1d = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=kernel_size, dilation=1)
        self.conv1d_dilated0 = CausalConv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=2)
        self.conv1d_dilated1 = CausalConv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=2**2)
        self.conv1d_dilated2 = CausalConv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, dilation=2**3)

        # conv output layers
        self.conv1d_out = CausalConv1d(in_channels=12 , out_channels=8, kernel_size=1)
        self.dropout_out = Dropout(dropout_rate)
        self.flatten_out = Flatten()

        # layers for concatenating with cat and num features
        self.dense_concat0 = Linear(in_features=8*args.use_days, out_features=56)
        self.dense_concat1 = Linear(in_features=56, out_features=n_outputs)
    
    def __call__(self, seq_in, t, criterion):
        pred = self.forward(seq_in)
        loss = criterion(pred, t)
        return loss, pred
    
    def forward(self, seq_in):
        # conv layers
        h0 = F.relu(self.conv1d(seq_in))
        h1 = F.relu(self.conv1d_dilated0(seq_in))
        h2 = F.relu(self.conv1d_dilated1(seq_in))
        h3 = F.relu(self.conv1d_dilated2(seq_in))

        # Skip connections
        c_ = torch.cat((h0, h1, h2, h3), 1)

        # out put of conv layers
        conv_out = F.relu(self.conv1d_out(c_))
        # conv_out = F.relu(self.dropout_out(conv_out))
        conv_out = F.relu(self.flatten_out(conv_out))

        # decode
        x = self.dense_concat0(conv_out)
        output = self.dense_concat1(x)

        return output.view(-1,9,28)

class kaggler_wavenet(nn.Module):
    def __init__(self, args, n_dyn_fea):
        super(kaggler_wavenet, self).__init__(
        )
        # params
        seq_len = args.use_days
        self.n_dyn_fea = n_dyn_fea
        self.n_dilated_layers = 3
        n_filters = 3
        n_outputs = 28
        dropout_rate = 0.1

        # Dilated convolutional layers
        self.conv1d = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=3, dilation=1)
        self.conv1d_dilated0 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=7, dilation=3)
        self.conv1d_dilated1 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=14, dilation=7)
        self.conv1d_dilated2 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=28, dilation=14)

        # layers for concatenating with cat and num features
        self.flatten = Flatten()
        self.dense_concat0 = Linear(in_features=n_filters*4*args.use_days, out_features=112)
        self.dropout_out = Dropout(0.2)
        self.dense_concat1 = Linear(in_features=112, out_features=n_outputs)
    
    def __call__(self, seq_in, t, criterion):
        pred = self.forward(seq_in)
        loss = criterion(pred, t)
        return loss, pred
    
    def forward(self, seq_in):
        # conv layers
        h0 = F.relu(self.conv1d(seq_in))
        h1 = F.relu(self.conv1d_dilated0(seq_in))
        h2 = F.relu(self.conv1d_dilated1(seq_in))
        h3 = F.relu(self.conv1d_dilated2(seq_in))
        c_ = torch.cat((h0, h1, h2, h3), 1)

        # decode
        x = self.dropout_out(self.dense_concat0(self.flatten(c_)))
        output = self.dense_concat1(x)

        return output

class amane_wavenet(nn.Module):
    def __init__(self, args, n_dyn_fea):
        super(amane_wavenet, self).__init__(
        )
        # params
        seq_len = args.use_days
        self.n_dyn_fea = n_dyn_fea
        n_filters = 64
        n_outputs = 28

        # Dilated convolutional layers
        self.conv1d = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=3, dilation=1)
        self.conv1d_dilated0 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=7, dilation=1)
        self.conv1d_dilated1 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=14, dilation=7)
        self.conv1d_dilated2 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=28, dilation=14)
        self.conv1d_dilated3 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=56, dilation=28)
        self.conv1d_dilated4 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=84, dilation=42)
        self.conv1d_dilated5 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=112, dilation=56)

        # layers for concatenating with cat and num features
        self.dropout_out = Dropout(0.05)
        self.dense_concat0 = Linear(in_features=7*n_filters, out_features=int(0.3*7*n_filters))
        self.dense_concat1 = Linear(in_features=int(0.3*7*n_filters), out_features=n_outputs)
    
    def __call__(self, seq_in, t, criterion):
        pred = self.forward(seq_in)
        loss = criterion(pred, t)
        return loss, pred
    
    def forward(self, seq_in):
        # conv layers
        h0,_ = torch.max(F.relu(self.conv1d(seq_in)),-1)
        h1,_ = torch.max(F.relu(self.conv1d_dilated0(seq_in)),-1)
        h2,_ = torch.max(F.relu(self.conv1d_dilated1(seq_in)),-1)
        h3,_ = torch.max(F.relu(self.conv1d_dilated2(seq_in)),-1)
        h4,_ = torch.max(F.relu(self.conv1d_dilated3(seq_in)),-1)
        h5,_ = torch.max(F.relu(self.conv1d_dilated4(seq_in)),-1)
        h6,_ = torch.max(F.relu(self.conv1d_dilated5(seq_in)),-1)

        # Skip connections
        c_ = torch.cat((h0, h1, h2, h3, h4, h5, h6), 1)

        # decode
        x = self.dropout_out(self.dense_concat0(c_))
        output = self.dense_concat1(x)

        return output

class amane_wavenet_v2(nn.Module):
    def __init__(self, args, n_dyn_fea):
        super(amane_wavenet_v2, self).__init__(
        )
        # params
        seq_len = args.use_days
        self.n_dyn_fea = n_dyn_fea
        n_filters = 32
        n_outputs = 28

        # Dilated convolutional layers
        self.conv1d = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=3, dilation=1)
        self.conv1d_dilated0 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=7, dilation=1)
        self.conv1d_dilated1 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=14, dilation=7)
        self.conv1d_dilated2 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=28, dilation=14)
        self.conv1d_dilated3 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=56, dilation=28)
        self.conv1d_dilated4 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=84, dilation=42)
        self.conv1d_dilated5 = CausalConv1d(in_channels=n_dyn_fea, out_channels=n_filters, kernel_size=112, dilation=56)

        # layers for concatenating with cat and num features
        self.dropout_out = Dropout(0)
        self.dense_concat0 = Linear(in_features=2*7*n_filters, out_features=int(0.3*7*n_filters))
        self.dense_concat1 = Linear(in_features=int(0.3*7*n_filters), out_features=n_outputs)
    
    def __call__(self, seq_in, t, criterion):
        pred = self.forward(seq_in)
        loss = criterion(pred, t)
        return loss, pred
    
    def forward(self, seq_in):
        # conv layers
        h0,_ = torch.max(torch.sigmoid(self.conv1d(seq_in)),-1)
        h1,_ = torch.max(torch.sigmoid(self.conv1d_dilated0(seq_in)),-1)
        h2,_ = torch.max(torch.sigmoid(self.conv1d_dilated1(seq_in)),-1)
        h3,_ = torch.max(torch.sigmoid(self.conv1d_dilated2(seq_in)),-1)
        h4,_ = torch.max(torch.sigmoid(self.conv1d_dilated3(seq_in)),-1)
        h5,_ = torch.max(torch.sigmoid(self.conv1d_dilated4(seq_in)),-1)
        h6,_ = torch.max(torch.sigmoid(self.conv1d_dilated5(seq_in)),-1)
        h7,_ = torch.min(torch.sigmoid(self.conv1d(seq_in)),-1)
        h8,_ = torch.min(torch.sigmoid(self.conv1d_dilated0(seq_in)),-1)
        h9,_ = torch.min(torch.sigmoid(self.conv1d_dilated1(seq_in)),-1)
        h10,_ = torch.min(torch.sigmoid(self.conv1d_dilated2(seq_in)),-1)
        h11,_ = torch.min(torch.sigmoid(self.conv1d_dilated3(seq_in)),-1)
        h12,_ = torch.min(torch.sigmoid(self.conv1d_dilated4(seq_in)),-1)
        h13,_ = torch.min(torch.sigmoid(self.conv1d_dilated5(seq_in)),-1)

        # Skip connections
        c_ = torch.cat((h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13), 1)

        # decode
        x = self.dropout_out(self.dense_concat0(c_))
        output = self.dense_concat1(x)

        return output
