__all__ = ['DualScalePatchTST_backbone']

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


class DualScalePatchTST_backbone(nn.Module):
    """
    Dual-Scale PatchTST Backbone
    Processes two patch sizes in parallel, concatenates them, then passes through Transformer
    """
    def __init__(self, c_in, context_window, target_window,
                 patch_len_fine=8, stride_fine=4,
                 patch_len_coarse=32, stride_coarse=16,
                 n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0,
                 padding_patch='end', individual=False,
                 revin=True, affine=True, subtract_last=False,
                 verbose=False, **kwargs):
        
        super().__init__()
        
        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Fine-scale patching
        self.patch_len_fine = patch_len_fine
        self.stride_fine = stride_fine
        self.padding_patch = padding_patch
        
        # Coarse-scale patching
        self.patch_len_coarse = patch_len_coarse
        self.stride_coarse = stride_coarse
        
        # Calculate number of patches
        patch_num_fine = int((context_window - patch_len_fine) / stride_fine + 1)
        if padding_patch == 'end':
            patch_num_fine += 1
        
        patch_num_coarse = int((context_window - patch_len_coarse) / stride_coarse + 1)
        if padding_patch == 'end':
            patch_num_coarse += 1
        
        self.patch_num_fine = patch_num_fine
        self.patch_num_coarse = patch_num_coarse
        total_patches = patch_num_fine + patch_num_coarse
        
        # Projection layers for each scale
        self.W_P_fine = nn.Linear(patch_len_fine, d_model)
        self.W_P_coarse = nn.Linear(patch_len_coarse, d_model)
        
        # Positional encoding for combined patches
        self.W_pos = positional_encoding('zeros', True, total_patches, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.encoder = TSTEncoder(
            q_len=total_patches,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation='gelu',
            res_attention=True,
            n_layers=n_layers,
            pre_norm=False,
            store_attn=False
        )
        
        # Head
        self.head_nf = d_model * total_patches
        self.n_vars = c_in
        self.individual = individual
        self.target_window = target_window
        
        if individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(self.head_nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self.head_nf, target_window)
            self.dropout_head = nn.Dropout(head_dropout)
    
    def forward(self, z):
        # z: [bs x nvars x seq_len]
        
        # RevIN normalization
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        
        # Create fine-scale patches with its own padding
        if self.padding_patch == 'end':
            padding_fine = nn.ReplicationPad1d((0, self.stride_fine))
            z_fine_padded = padding_fine(z)
            z_fine = z_fine_padded.unfold(dimension=-1, size=self.patch_len_fine, step=self.stride_fine)
        else:
            z_fine = z.unfold(dimension=-1, size=self.patch_len_fine, step=self.stride_fine)
        z_fine = z_fine.permute(0, 1, 3, 2)  # [bs, nvars, patch_len, patch_num]
        
        # Create coarse-scale patches with its own padding
        if self.padding_patch == 'end':
            padding_coarse = nn.ReplicationPad1d((0, self.stride_coarse))
            z_coarse_padded = padding_coarse(z)
            z_coarse = z_coarse_padded.unfold(dimension=-1, size=self.patch_len_coarse, step=self.stride_coarse)
        else:
            z_coarse = z.unfold(dimension=-1, size=self.patch_len_coarse, step=self.stride_coarse)
        z_coarse = z_coarse.permute(0, 1, 3, 2)  # [bs, nvars, patch_len, patch_num]
        
        # Project both scales to d_model
        z_fine = z_fine.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, patch_len]
        z_coarse = z_coarse.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, patch_len]
        
        x_fine = self.W_P_fine(z_fine)  # [bs, nvars, patch_num_fine, d_model]
        x_coarse = self.W_P_coarse(z_coarse)  # [bs, nvars, patch_num_coarse, d_model]
        
        # Concatenate patches from both scales
        x = torch.cat([x_fine, x_coarse], dim=2)  # [bs, nvars, total_patches, d_model]
        
        # Reshape for Transformer
        bs = x.shape[0]
        nvars = x.shape[1]
        u = x.reshape(bs * nvars, -1, x.shape[-1])  # [bs*nvars, total_patches, d_model]
        
        # Add positional encoding (with safety check)
        if u.shape[1] != self.W_pos.shape[0]:
            # Pad or trim to match
            if u.shape[1] < self.W_pos.shape[0]:
                padding = torch.zeros(u.shape[0], self.W_pos.shape[0] - u.shape[1], u.shape[2]).to(u.device)
                u = torch.cat([u, padding], dim=1)
            else:
                u = u[:, :self.W_pos.shape[0], :]
        
        u = self.dropout(u + self.W_pos)
        
        # Transformer encoder
        z = self.encoder(u)  # [bs*nvars, total_patches, d_model]
        
        # Reshape back
        z = z.reshape(bs, nvars, -1, z.shape[-1])  # [bs, nvars, total_patches, d_model]
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, d_model, total_patches]
        
        # Head
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                flat = self.flattens[i](z[:, i, :, :])
                out = self.linears[i](flat)
                out = self.dropouts[i](out)
                x_out.append(out)
            z = torch.stack(x_out, dim=1)
        else:
            z = self.flatten(z)
            z = self.linear(z)
            z = self.dropout_head(z)
        
        # RevIN denormalization
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        
        return z


class TSTEncoder(nn.Module):
    """Transformer Encoder with residual attention"""
    def __init__(self, q_len, d_model, n_heads, d_ff=256, dropout=0.1, activation='gelu',
                 res_attention=True, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads, d_ff, dropout, activation, res_attention, pre_norm, store_attn)
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
    """Single Transformer Encoder Layer"""
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1, activation='gelu',
                 res_attention=False, pre_norm=False, store_attn=False):
        super().__init__()
        
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        
        # Self-attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, dropout, res_attention)
        
        # Normalization
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.res_attention = res_attention
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
        
        # Feed-forward
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


class _MultiheadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, d_model, n_heads, d_k, d_v, dropout=0.1, res_attention=False):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(dropout)
        )
        
        self.scale = d_k ** -0.5
        self.res_attention = res_attention
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, prev=None):
        bs = Q.size(0)
        
        # Linear projections
        q = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k) * self.scale
        
        if prev is not None:
            attn_scores = attn_scores + prev
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights