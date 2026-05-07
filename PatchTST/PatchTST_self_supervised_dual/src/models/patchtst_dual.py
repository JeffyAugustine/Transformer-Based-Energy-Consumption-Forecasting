__all__ = ['DualScalePatchTST']

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math

from ..models.layers.pos_encoding import positional_encoding
from ..models.layers.basics import Transpose, get_activation_fn
from ..models.layers.attention import MultiheadAttention


class DualScalePatchTST(nn.Module):
    """
    Dual-Scale PatchTST for Self-Supervised Learning
    Processes two parallel patch streams: fine (P=8, stride=4) and coarse (P=32, stride=16)
    """
    def __init__(self, c_in: int, target_dim: int,
                 patch_len_fine: int = 8, stride_fine: int = 4,
                 patch_len_coarse: int = 32, stride_coarse: int = 16,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16,
                 shared_embedding: bool = True, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 head_dropout: float = 0, head_type: str = "pretrain",
                 individual: bool = False, y_range: tuple = None,
                 verbose: bool = False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], \
            f'head type should be either pretrain, prediction, or regression, got {head_type}'

        self.c_in = c_in
        self.d_model = d_model
        self.patch_len_fine = patch_len_fine
        self.stride_fine = stride_fine
        self.patch_len_coarse = patch_len_coarse
        self.stride_coarse = stride_coarse
        self.shared_embedding = shared_embedding
        self.dropout = nn.Dropout(dropout)

        # Separate projection layers for fine and coarse scales
        if not shared_embedding:
            self.W_P_fine = nn.ModuleList([nn.Linear(patch_len_fine, d_model) for _ in range(c_in)])
            self.W_P_coarse = nn.ModuleList([nn.Linear(patch_len_coarse, d_model) for _ in range(c_in)])
        else:
            self.W_P_fine = nn.Linear(patch_len_fine, d_model)
            self.W_P_coarse = nn.Linear(patch_len_coarse, d_model)

        # Shared Transformer Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act,
                                  res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

        # Head
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = DualScalePretrainHead(d_model, patch_len_fine, patch_len_coarse, head_dropout)
        elif head_type == "prediction":
            self.head = DualScalePredictionHead(individual, self.n_vars, d_model, head_dropout, target_dim)
        elif head_type == "regression":
            self.head = DualScaleRegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = DualScaleClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

    def forward(self, x):
        """
        x: tuple (z_fine, z_coarse) from callback
        Expected shape from callback: [bs, num_patches, n_vars, patch_len]
        """
        patch_len_fine = self.patch_len_fine
        stride_fine = self.stride_fine
        patch_len_coarse = self.patch_len_coarse
        stride_coarse = self.stride_coarse
        if isinstance(x, (tuple, list)):
            z_fine = x[0]  # [bs, num_patch_fine, n_vars, patch_len_fine]
            z_coarse = x[1]  # [bs, num_patch_coarse, n_vars, patch_len_coarse]
        else:
            # Single tensor - during lr_finder
            z_fine = x.unfold(dimension=1, size=patch_len_fine, step=stride_fine)
            z_fine = z_fine.permute(0, 1, 3, 2)
            z_coarse = x.unfold(dimension=1, size=patch_len_coarse, step=stride_coarse)
            z_coarse = z_coarse.permute(0, 1, 3, 2)
        
        
        # Get dimensions
        bs = z_fine.shape[0]
        n_vars = z_fine.shape[2]  # shape[2] is n_vars (7)
        num_patch_fine = z_fine.shape[1]
        num_patch_coarse = z_coarse.shape[1]
        total_patches = num_patch_fine + num_patch_coarse

        # Set patch counts on the head (for pretrain head)
        if hasattr(self.head, 'set_num_patches'):
            self.head.set_num_patches(num_patch_fine, num_patch_coarse)

        # Positional encoding for combined patches
        W_pos = positional_encoding('zeros', True, total_patches, self.d_model).to(z_fine.device)

        # Input encoding for fine scale
        if self.shared_embedding:
            # z_fine is [bs, num_patch_fine, n_vars, patch_len_fine]
            # Permute to [bs, n_vars, num_patch_fine, patch_len_fine]
            z_fine_reshaped = z_fine.permute(0, 2, 1, 3).reshape(-1, patch_len_fine)
            x_fine_enc = self.W_P_fine(z_fine_reshaped)
            x_fine_enc = x_fine_enc.reshape(bs, n_vars, num_patch_fine, self.d_model)
        else:
            x_fine_out = []
            for i in range(n_vars):
                x_fine_out.append(self.W_P_fine[i](z_fine[:, :, i, :]))
            x_fine_enc = torch.stack(x_fine_out, dim=2)

        # Input encoding for coarse scale
        if self.shared_embedding:
            # z_coarse is [bs, num_patch_coarse, n_vars, patch_len_coarse]
            # Permute to [bs, n_vars, num_patch_coarse, patch_len_coarse]
            z_coarse_reshaped = z_coarse.permute(0, 2, 1, 3).reshape(-1, patch_len_coarse)
            x_coarse_enc = self.W_P_coarse(z_coarse_reshaped)
            x_coarse_enc = x_coarse_enc.reshape(bs, n_vars, num_patch_coarse, self.d_model)
        else:
            x_coarse_out = []
            for i in range(n_vars):
                x_coarse_out.append(self.W_P_coarse[i](z_coarse[:, :, i, :]))
            x_coarse_enc = torch.stack(x_coarse_out, dim=2)

        # Concatenate patches from both scales: [bs, n_vars, total_patches, d_model]
        x_combined = torch.cat([x_fine_enc, x_coarse_enc], dim=2)
        
        # Reshape for Transformer: [bs * n_vars, total_patches, d_model]
        u = x_combined.reshape(bs * n_vars, total_patches, self.d_model)
        u = self.dropout(u + W_pos)

        # Encoder
        z = self.encoder(u)
        z = z.reshape(bs, n_vars, total_patches, self.d_model)
        z = z.permute(0, 1, 3, 2)  # [bs, n_vars, d_model, total_patches]

        # Head
        z = self.head(z)
        return z


class DualScalePretrainHead(nn.Module):
    def __init__(self, d_model, patch_len_fine, patch_len_coarse, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_fine = nn.Linear(d_model, patch_len_fine)
        self.linear_coarse = nn.Linear(d_model, patch_len_coarse)
        self.num_patch_fine = None
        self.num_patch_coarse = None

    def set_num_patches(self, num_patch_fine, num_patch_coarse):
        self.num_patch_fine = num_patch_fine
        self.num_patch_coarse = num_patch_coarse

    def forward(self, x):
        """
        x: [bs x nvars x d_model x total_patches]
        returns: tuple (fine_patches, coarse_patches)
        """
        if self.num_patch_fine is None:
            raise ValueError("set_num_patches() must be called before forward")
        
        x = x.transpose(2, 3)  # [bs x nvars x total_patches x d_model]
        
        x_fine = x[:, :, :self.num_patch_fine, :]
        x_coarse = x[:, :, self.num_patch_fine:, :]
        
        x_fine = self.linear_fine(self.dropout(x_fine))
        x_coarse = self.linear_coarse(self.dropout(x_coarse))
        
        # Reshape to [bs, num_patches, nvars, patch_len]
        x_fine = x_fine.permute(0, 2, 1, 3)
        x_coarse = x_coarse.permute(0, 2, 1, 3)
        
        return (x_fine, x_coarse)


class DualScalePredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, head_dropout, forecast_len):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.forecast_len = forecast_len
        self.linear = None

    def forward(self, x):
        bs, n_vars, d_model, total_patches = x.shape
        head_dim = d_model * total_patches
        
        if self.linear is None or self.linear.in_features != head_dim:
            self.linear = nn.Linear(head_dim, self.forecast_len).to(x.device)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x.transpose(1, 2)


class DualScaleRegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, output_dim)

    def forward(self, x):
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        if self.y_range:
            y = torch.sigmoid(y) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return y


class DualScaleClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.linear(x)
        return y


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                            attn_dropout=attn_dropout, dropout=dropout,
                            activation=activation, res_attention=res_attention,
                            pre_norm=pre_norm, store_attn=store_attn)
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers:
                output = mod(output)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src, prev=None):
        if self.pre_norm:
            src = self.norm_attn(src)

        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)

        if self.store_attn:
            self.attn = attn

        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)

        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src