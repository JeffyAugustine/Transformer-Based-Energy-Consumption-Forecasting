import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchtst_dual import DualScalePatchTST
from src.learner import Learner
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import get_dls

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Dual-Scale Patch
parser.add_argument('--patch_len_fine', type=int, default=8, help='fine patch length')
parser.add_argument('--stride_fine', type=int, default=4, help='fine stride')
parser.add_argument('--patch_len_coarse', type=int, default=32, help='coarse patch length')
parser.add_argument('--stride_coarse', type=int, default=16, help='coarse stride')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Transformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')

args = parser.parse_args()
print('args:', args)

# Calculate number of patches
num_patch_fine = (max(args.context_points, args.patch_len_fine) - args.patch_len_fine) // args.stride_fine + 1
num_patch_coarse = (max(args.context_points, args.patch_len_coarse) - args.patch_len_coarse) // args.stride_coarse + 1
print(f'Number of fine patches: {num_patch_fine}')
print(f'Number of coarse patches: {num_patch_coarse}')
print(f'Total patches: {num_patch_fine + num_patch_coarse}')

args.save_pretrained_model = f'patchtst_dual_pretrained_cw{args.context_points}_fineP{args.patch_len_fine}_coarseP{args.patch_len_coarse}_mask{args.mask_ratio}_model{args.pretrained_model_id}'
args.save_path = f'saved_models/{args.dset_pretrain}/masked_patchtst_dual/{args.model_type}/'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

set_device()


def get_model(c_in, args):
    model = DualScalePatchTST(
        c_in=c_in,
        target_dim=args.target_points,
        patch_len_fine=args.patch_len_fine,
        stride_fine=args.stride_fine,
        patch_len_coarse=args.patch_len_coarse,
        stride_coarse=args.stride_coarse,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='pretrain',
        res_attention=False
    )
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    loss_func = torch.nn.MSELoss(reduction='mean')
    cbs = []
    cbs += [DualScalePatchMaskCB(patch_len_fine=args.patch_len_fine, stride_fine=args.stride_fine,
                                  patch_len_coarse=args.patch_len_coarse, stride_coarse=args.stride_coarse,
                                  mask_ratio=args.mask_ratio)]

    learn = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs)
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    loss_func = torch.nn.MSELoss(reduction='mean')
    cbs = []
    cbs += [
        DualScalePatchMaskCB(patch_len_fine=args.patch_len_fine, stride_fine=args.stride_fine,
                              patch_len_coarse=args.patch_len_coarse, stride_coarse=args.stride_coarse,
                              mask_ratio=args.mask_ratio),
        SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model, path=args.save_path)
    ]
    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs)
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    args.dset = args.dset_pretrain
    pretrain_func(args.lr)
    print('pretraining completed')