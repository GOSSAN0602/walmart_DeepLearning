import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class simple_Net(nn.Module):
    def __init__(self):
        super(simple_Net, self).__init__()
        self.layer1 = nn.LSTM(1,3,1,True,False,0.1,False)
        self.h0 = torch.randn(256, 1, 3)
        self.c0 = torch.randn(256, 1, 3)
    
    def __call__(self, x, t, criterion):
        steps = t.shape[1]
        (hn, cn) = self.open_loop(x)
        loss, pred_seq = self.closed_loop(x[-1], hn, cn, t, steps, criterion)


    def open_loop(self, x):
        import pdb;pdb.set_trace()
        _, (hn, cn) = self.layer1(x[0], (self.h0, self.c0))
        for i in range(x.shape[1]-1):
            _, (hn, cn) = self.layer1(x[i+1], (hn, cn))
        return (hn, cn)
    
    def closed_loop(self, x, hn, cn, t, steps, criterion):
        pred_seq = np.zeros([t.shape[0], steps])
        loss = 0.0
        _x, (hn, cn) = self.layer1(x, (hn, cn))
        loss += criterion(_x, t[0])
        pred_seq[0] = _x.item()
        for i in range(steps-1):
            _x, (hn, cn) = self.layer1(_x, (hn, cn))
            loss += criterion(_x, t[i+1])
            pred_seq[i+1] = _x.item()
        return loss, pred_seq