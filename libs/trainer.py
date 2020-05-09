import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

def set_optimizer(args, model):
    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr = args.lr)
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr = args.lr, betas=(args.beta1,args.beta2))
    if args.optimizer == 'RAdam':
        return RAdam(model.parameters(), lr = args.lr, betas=(args.beta1,args.beta2))


def rnn_trainer(args, model, tr_x, tr_t, va_x, va_t, log_dir):
    # config for train NN
    n_iter = int(tr_x.shape[0] / args.batch_size)+1
    batch_idx = np.arange(tr_x.shape[0])
    pred_seq = np.zeros(tr_t.shape)
    interval = args.interval
    batch_size = args.batch_size

    optimizer = set_optimizer(args, model)
    cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
    criterion = nn.MSELoss()
    loss_tr = np.zeros(int(args.n_epoch/interval))
    loss_va = np.zeros(int(args.n_epoch/interval))

    # epoch
    for i in range(args.n_epoch):
        batch_idx = np.random.permutation(batch_idx)
        model.train()
        for iter in range(n_iter-1):
            # initialize optimizer
            optimizer.zero_grad()
            use_idx = batch_idx[iter*batch_size:(iter+1)*batch_size]
            batch_x = tr_x[use_idx]
            batch_t = tr_t[use_idx]
            loss, _ = model(batch_x, batch_t, criterion)
            loss.backward()
            optimizer.step()
            print(loss.item())
        cos_lr_scheduler.step()
        if (i+1) % interval ==0:
            model.eval()
            loss_tr[int(i/interval)] = loss.item()
            va_loss, pred_seq = model(va_x, va_t, criterion)
            loss_va[int(i/interval)] = va_loss.item()
            #plot_loss(loss_tr,loss_va)
            if loss_va[int(i/interval)] <= loss_va[:(1+int(i/interval))].min():
                print(f'epoch: {i+1}  score improved  {loss_va[int(i/interval)]}')
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                ax.plot(np.arange(interval, interval+i+1,10),loss_tr[:int((i+1)/interval)], label="train")
                ax.plot(np.arange(interval, interval+i+1,10),loss_va[:int((i+1)/interval)], label="valid")
                ax.legend()
                plt.savefig(log_dir+'/loss_curve.png')
                np.save(log_dir+'/pred_valid.npy', pred_seq)