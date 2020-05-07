import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class simple_Net(nn.Module):
    def __init__(self, args):
        super(simple_Net, self).__init__()
        self.layer1 = nn.LSTM(1,1,1,True,False,0.1,False)
    
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
        loss += criterion(_x, t[:,:,0:1])
        pred_seq[:,0:1] = _x.detach().numpy()
        for i in range(steps-1):
            _x, (hn, cn) = self.layer1(_x, (hn, cn))
            loss += criterion(_x, t[:,:,(i+1):(i+2)].reshape(1,-1,1))
            pred_seq[:,(i+1):(i+2)] = _x.detach().numpy()
        return loss, pred_seq