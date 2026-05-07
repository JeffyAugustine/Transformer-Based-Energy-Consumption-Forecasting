import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchtst_dual import DualScalePatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import get_dls

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset
parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
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
# Optimization
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')

args = parser.parse_args()
print('args:', args)

num_patch_fine = (max(args.context_points, args.patch_len_fine) - args.patch_len_fine) // args.stride_fine + 1
num_patch_coarse = (max(args.context_points, args.patch_len_coarse) - args.patch_len_coarse) // args.stride_coarse + 1
print(f'Number of fine patches: {num_patch_fine}')
print(f'Number of coarse patches: {num_patch_coarse}')
print(f'Total patches: {num_patch_fine + num_patch_coarse}')

args.save_path = f'saved_models/{args.dset_finetune}/masked_patchtst_dual/{args.model_type}/'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

suffix_name = f'_cw{args.context_points}_tw{args.target_points}_fineP{args.patch_len_fine}_coarseP{args.patch_len_coarse}_epochs{args.n_epochs_finetune}_model{args.finetuned_model_id}'
if args.is_finetune:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_dual_finetuned' + suffix_name
elif args.is_linear_probe:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_dual_linear-probe' + suffix_name
else:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_dual_finetuned' + suffix_name

set_device()


def get_model(c_in, args, head_type, weight_path=None):
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
        head_type=head_type,
        res_attention=False
    )
    if weight_path:
        model = transfer_weights(weight_path, model, exclude_head=True)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction', weight_path=args.pretrained_model)
    loss_func = torch.nn.MSELoss(reduction='mean')
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
        DualScalePatchCB(patch_len_fine=args.patch_len_fine, stride_fine=args.stride_fine,
                          patch_len_coarse=args.patch_len_coarse, stride_coarse=args.stride_coarse),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
    ]
    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs, metrics=[mse])
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)


def test_func(weight_path):
    print("STARTING TEST FUNCTION")
    
    dls = get_dls(args)
    
    model = get_model(dls.vars, args, head_type='prediction')
    
    load_path = weight_path + '.pth'
    state_dict = torch.load(load_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()
    
    print(f" Model loaded successfully")
    
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [DualScalePatchCB(patch_len_fine=args.patch_len_fine, stride_fine=args.stride_fine,
                              patch_len_coarse=args.patch_len_coarse, stride_coarse=args.stride_coarse)]
    learn = Learner(dls, model, cbs=cbs)
    
    print(" Running test...")
    out = learn.test(dls.test, scores=[mse, mae])
    
    # Analyze predictions vs targets
    preds = out[0]
    targets = out[1]
    
    print(f"\n Statistical analysis:")
    print(f" Predictions - min: {preds.min():.6f}, max: {preds.max():.6f}, mean: {preds.mean():.6f}, std: {preds.std():.6f}")
    print(f" Targets - min: {targets.min():.6f}, max: {targets.max():.6f}, mean: {targets.mean():.6f}, std: {targets.std():.6f}")
    
    # Check first sample's OT predictions (index 0, all time steps, OT feature at index 0)
    print(f"\n First sample - OT feature (first 20 steps):")
    print(f" Predictions: {preds[0, :20, 0]}")
    print(f" Targets: {targets[0, :20, 0]}")
    
    # Check if predictions are constant
    pred_unique = len(np.unique(preds[:100]))
    print(f"\n Unique values in first 100 predictions: {pred_unique}")
    if pred_unique < 10:
        print(" WARNING: Predictions are almost constant! Model not learning.")
    
    # Check if predictions are in normalized space
    if preds.mean() < 1 and preds.std() < 1:
        print(" Predictions appear to be in normalized space (mean near 0, std near 1)")
    
    print('score:', out[2])
    
    # Save with debug info in filename
    debug_csv_path = args.save_path + args.save_finetuned_model + '_acc_debug.csv'
    pd.DataFrame(np.array(out[2]).reshape(1, -1), columns=['mse', 'mae']).to_csv(
        debug_csv_path, float_format='%.6f', index=False)
    print(f" Results saved to: {debug_csv_path}")
    
    # Also save predictions for analysis
    np.save(args.save_path + args.save_finetuned_model + '_predictions.npy', preds)
    np.save(args.save_path + args.save_finetuned_model + '_targets.npy', targets)
    
    return out


if __name__ == '__main__':
    if args.is_finetune:
        args.dset = args.dset_finetune
        finetune_func(args.lr)
        print('finetune completed')
        out = test_func(args.save_path + args.save_finetuned_model)