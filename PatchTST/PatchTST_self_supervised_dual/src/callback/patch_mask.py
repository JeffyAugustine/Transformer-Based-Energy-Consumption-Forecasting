import torch
from torch import nn
from .core import Callback


class PatchCB(Callback):
    def __init__(self, patch_len, stride):
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self):
        self.set_patch()
       
    def set_patch(self):
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)
        self.learner.xb = xb_patch


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio, mask_when_pred: bool = False):
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio

    def before_fit(self):
        self.learner.loss_func = self._loss

    def before_forward(self):
        self.patch_masking()
        
    def patch_masking(self):
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)
        self.mask = self.mask.bool()
        self.learner.xb = xb_mask
        self.learner.yb = xb_patch
 
    def _loss(self, preds, target):
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len
    xb = xb[:, s_begin:, :]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return x


def random_masking(xb, mask_ratio):
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(bs, L, nvars, device=xb.device)
    
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep, :]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))
    
    x_removed = torch.zeros(bs, L - len_keep, nvars, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))
    
    mask = torch.ones([bs, L, nvars], device=x.device)
    mask[:, :len_keep, :] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(bs, L, device=xb.device)
    
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, x_kept, mask, ids_restore


class DualScalePatchMaskCB(Callback):
    """
    Dual-scale masking for pre-training.
    """
    
    def __init__(self, patch_len_fine=8, stride_fine=4, 
                 patch_len_coarse=32, stride_coarse=16, mask_ratio=0.4):
        super().__init__()
        self.patch_len_fine = patch_len_fine
        self.stride_fine = stride_fine
        self.patch_len_coarse = patch_len_coarse
        self.stride_coarse = stride_coarse
        self.mask_ratio = mask_ratio

    def before_fit(self):
        self.learner.loss_func = self._loss

    def before_forward(self):
        self.patch_masking()
        
    def patch_masking(self):
        x = self.learner.xb  # [bs, seq_len, n_vars]
        
        # unfold gives [bs, num_patches, n_vars, patch_len] - THIS IS CORRECT!
        x_fine = x.unfold(dimension=1, size=self.patch_len_fine, step=self.stride_fine)
        x_coarse = x.unfold(dimension=1, size=self.patch_len_coarse, step=self.stride_coarse)
        
        # Apply random masking
        x_fine_masked, _, self.mask_fine, _ = random_masking(x_fine, self.mask_ratio)
        x_coarse_masked, _, self.mask_coarse, _ = random_masking(x_coarse, self.mask_ratio)
        
        self.mask_fine = self.mask_fine.bool()
        self.mask_coarse = self.mask_coarse.bool()
        
        # Store targets and masked input
        self.learner.xb = (x_fine_masked, x_coarse_masked)
        self.learner.yb = (x_fine, x_coarse)
    
    def _loss(self, preds, target):
        pred_fine, pred_coarse = preds
        target_fine, target_coarse = self.learner.yb
        
        loss_fine = (pred_fine - target_fine) ** 2
        loss_fine = loss_fine.mean(dim=-1)
        loss_fine = (loss_fine * self.mask_fine).sum() / (self.mask_fine.sum() + 1e-8)
        
        loss_coarse = (pred_coarse - target_coarse) ** 2
        loss_coarse = loss_coarse.mean(dim=-1)
        loss_coarse = (loss_coarse * self.mask_coarse).sum() / (self.mask_coarse.sum() + 1e-8)
        
        return loss_fine + loss_coarse


class DualScalePatchCB(Callback):
    def __init__(self, patch_len_fine=8, stride_fine=4, patch_len_coarse=32, stride_coarse=16):
        super().__init__()
        self.patch_len_fine = patch_len_fine
        self.stride_fine = stride_fine
        self.patch_len_coarse = patch_len_coarse
        self.stride_coarse = stride_coarse

    def before_forward(self):
        if self.learner.xb is None:
            return
        
        x = self.learner.xb
        if isinstance(x, tuple):
            x = x[0]
        if x is None or len(x.shape) != 3:
            return
        
        x_fine = x.unfold(dimension=1, size=self.patch_len_fine, step=self.stride_fine)
        x_coarse = x.unfold(dimension=1, size=self.patch_len_coarse, step=self.stride_coarse)
        
        
        self.learner.xb = (x_fine, x_coarse)


if __name__ == "__main__":
    bs, L, nvars, D = 2, 20, 4, 5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, _, mask, _ = random_masking(xb, mask_ratio=0.5)