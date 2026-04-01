__all__ = ['DualScalePatchTST']

import torch
import torch.nn as nn
import numpy as np
from layers.PatchTST_backbone_dual import DualScalePatchTST_backbone
from layers.PatchTST_layers import series_decomp


class DualScalePatchTST(nn.Module):
    """
    Dual-Scale PatchTST Model
    Uses two parallel patch sizes: fine (P=8) and coarse (P=32)
    Concatenates patches from both scales before the Transformer encoder
    """
    def __init__(self, configs):
        super().__init__()
        
        # Load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # Dual-scale parameters
        patch_len_fine = getattr(configs, 'patch_len_fine', 8)
        stride_fine = getattr(configs, 'stride_fine', 4)
        patch_len_coarse = getattr(configs, 'patch_len_coarse', 32)
        stride_coarse = getattr(configs, 'stride_coarse', 16)
        padding_patch = configs.padding_patch
        
        # Model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = DualScalePatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len_fine=patch_len_fine, stride_fine=stride_fine,
                patch_len_coarse=patch_len_coarse, stride_coarse=stride_coarse,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch, individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=False
            )
            self.model_res = DualScalePatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len_fine=patch_len_fine, stride_fine=stride_fine,
                patch_len_coarse=patch_len_coarse, stride_coarse=stride_coarse,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch, individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=False
            )
        else:
            self.model = DualScalePatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len_fine=patch_len_fine, stride_fine=stride_fine,
                patch_len_coarse=patch_len_coarse, stride_coarse=stride_coarse,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout,
                padding_patch=padding_patch, individual=individual, revin=revin, affine=affine,
                subtract_last=subtract_last, verbose=False
            )
    
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)  # [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # [Batch, Input length, Channel]
        return x