import torch
import numpy as np
import time
import os
import sys
import argparse

sys.path.insert(0, './src')
from src.models.patchTST import PatchTST
from src.callback.patch_mask import PatchCB
from src.callback.transforms import RevInCB
from src.metrics import mse, mae
from src.learner import Learner
from datautils import get_dls

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_only', action='store_true', help='Save predictions without displaying')
    args = parser.parse_args()
    
    save_mode = args.save_only
    
    if not save_mode:
        print("DEMO: Self-Supervised Single-Scale PatchTST")
    
    # Base directory for saving predictions
    base_save_dir = '/scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting/demo_predictions'
    model_save_dir = os.path.join(base_save_dir, 'self_single')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Data loading args
    class DataArgs:
        pass
    
    data_args = DataArgs()
    data_args.dset = 'etth1'
    data_args.context_points = 336
    data_args.target_points = 96
    data_args.batch_size = 32
    data_args.num_workers = 4
    data_args.scaler = 'standard'
    data_args.features = 'M'
    data_args.use_time_features = False
    data_args.patch_len = 16
    data_args.stride = 8
    data_args.revin = 1
    
    f_dim = -1 if data_args.features == 'MS' else 0
    
    # Horizons to evaluate
    horizons = [96, 192, 336, 720]
    
    # Checkpoint paths for each horizon (update paths as needed)
    checkpoint_paths = {
        96: 'saved_models/etth1/masked_patchtst/based_model/etth1_patchtst_finetuned_cw336_tw96_patch16_stride8_epochs-finetune20_model1.pth',
        192: 'saved_models/etth1/masked_patchtst/based_model/etth1_patchtst_finetuned_cw336_tw192_patch16_stride8_epochs-finetune20_model1.pth',
        336: 'saved_models/etth1/masked_patchtst/based_model/etth1_patchtst_finetuned_cw336_tw336_patch16_stride8_epochs-finetune20_model1.pth',
        720: 'saved_models/etth1/masked_patchtst/based_model/etth1_patchtst_finetuned_cw336_tw720_patch16_stride8_epochs-finetune20_model1.pth',
    }
    
    results = []
    
    if not save_mode:
        print("\n[1] Running inference for all horizons...")
    
    for horizon in horizons:
        if not save_mode:
            print(f"\n    --- Horizon: {horizon} hours ---")
        
        # Update target points
        data_args.target_points = horizon
        
        # Load test data for this horizon
        dls = get_dls(data_args)
        
        # Get one test sample
        for batch_x, batch_y in dls.test:
            demo_input = batch_x
            demo_target = batch_y
            break
        
        # Save ground truth
        gt_np = demo_target.numpy()
        np.save(os.path.join(base_save_dir, f'ground_truth_t{horizon}.npy'), gt_np)
        
        # Load model
        num_patch = (336 - 16) // 8 + 1
        model = PatchTST(
            c_in=7,
            target_dim=horizon,
            patch_len=16,
            stride=8,
            num_patch=num_patch,
            n_layers=3,
            n_heads=4,
            d_model=16,
            d_ff=128,
            dropout=0.2,
            head_dropout=0.2,
            head_type='prediction'
        ).cuda()
        
        checkpoint_path = checkpoint_paths[horizon]
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'), strict=False)
        else:
            if not save_mode:
                print(f"        WARNING: Checkpoint not found: {checkpoint_path}")
            results.append((horizon, -1, -1, -1))  # Placeholder for missing
            continue
        
        model.eval()
        
        # Run inference using Learner approach
        cbs = [RevInCB(dls.vars, denorm=True)] if data_args.revin else []
        cbs += [PatchCB(patch_len=data_args.patch_len, stride=data_args.stride)]
        
        learn = Learner(dls, model, cbs=cbs)
        
        start = time.time()
        out = learn.test(dls.test, scores=[mse, mae])
        infer_time = (time.time() - start) * 1000
        
        # Get results
        pred_np = out[0]
        target_np = out[1]
        test_mse = out[2][0]
        test_mae = out[2][1]
        
        results.append((horizon, test_mse, test_mae, infer_time))
        
        # Save predictions
        np.save(os.path.join(model_save_dir, f't{horizon}_pred.npy'), pred_np)
        
        # Save metrics
        metrics = np.array([[test_mse, test_mae]])
        np.savetxt(os.path.join(model_save_dir, f't{horizon}_metrics.csv'), metrics, 
                   header='mse,mae', delimiter=',', comments='')
        
        if not save_mode:
            print(f"        MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, Time: {infer_time:.2f}ms")
            print(f"        Saved to: {model_save_dir}/t{horizon}_pred.npy")
    
    # Print summary table
    if not save_mode:
        print("SUMMARY - Self-Supervised Single-Scale")
        print(f"{'Horizon':<10} {'MSE':<12} {'MAE':<12} {'Time(ms)':<10}")
        print("-"*44)
        for h, mse_val, mae_val, t in results:
            if mse_val >= 0:
                print(f"{h:<10} {mse_val:.6f}   {mae_val:.6f}   {t:.2f}")
            else:
                print(f"{h:<10} {'NOT FOUND':<12} {'NOT FOUND':<12} {t:.2f}")
        print(f"All predictions saved to: {model_save_dir}")
    else:
        # Print minimal output for orchestrator
        for h, mse_val, mae_val, t in results:
            if mse_val >= 0:
                print(f"MSE: {mse_val:.6f}, MAE: {mae_val:.6f}, Horizon: {h}")

if __name__ == "__main__":
    run_demo()