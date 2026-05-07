import torch
import numpy as np
import time
import os
import argparse
from data_provider.data_factory import data_provider
from models.PatchTST_dual import DualScalePatchTST

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_only', action='store_true', help='Save predictions without displaying')
    args = parser.parse_args()
    
    save_mode = args.save_only
    
    if not save_mode:
        print("DEMO: Supervised Dual-Scale PatchTST (Our Novelty)")
    
    # Base directory for saving predictions
    base_save_dir = '/scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting/demo_predictions'
    model_save_dir = os.path.join(base_save_dir, 'supervised_dual')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Setup config (same for all horizons)
    class Config:
        pass
    
    args_config = Config()
    args_config.seq_len = 336
    args_config.enc_in = 7
    args_config.dec_in = 7
    args_config.c_out = 7
    args_config.individual = False
    args_config.revin = True
    args_config.affine = False
    args_config.subtract_last = False
    args_config.decomposition = False
    args_config.kernel_size = 25
    args_config.padding_patch = 'end'
    args_config.n_heads = 8
    args_config.d_model = 128
    args_config.d_ff = 256
    args_config.e_layers = 3
    args_config.dropout = 0.2
    args_config.fc_dropout = 0.2
    args_config.head_dropout = 0.0
    args_config.patch_len_fine = 8
    args_config.stride_fine = 4
    args_config.patch_len_coarse = 32
    args_config.stride_coarse = 16
    args_config.embed = 'timeF'
    args_config.freq = 'h'
    args_config.features = 'M'
    args_config.target = 'OT'
    args_config.output_attention = False
    args_config.data = 'ETTh1'
    args_config.root_path = '../../data/raw/'
    args_config.data_path = 'ETTh1.csv'
    args_config.batch_size = 1
    args_config.num_workers = 0
    args_config.label_len = 48
    
    f_dim = -1 if args_config.features == 'MS' else 0
    
    # Horizons and their checkpoint paths
    horizons = [96, 192, 336, 720]
    
    # Checkpoint paths for each horizon (update these paths as needed)
    checkpoint_paths = {
        96: 'checkpoints/ETTh1_336_96_dual_full_DualScalePatchTST_ETTh1_ftM_sl336_ll48_pl96_dm128_nh8_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_2/checkpoint.pth',
        192: 'checkpoints/ETTh1_336_192_dual_DualScalePatchTST_ETTh1_ftM_sl336_ll48_pl192_dm128_nh8_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth',
        336: 'checkpoints/ETTh1_336_336_dual_DualScalePatchTST_ETTh1_ftM_sl336_ll48_pl336_dm128_nh8_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth',
        720: 'checkpoints/ETTh1_336_720_dual_DualScalePatchTST_ETTh1_ftM_sl336_ll48_pl720_dm128_nh8_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth',
    }
    
    results = []
    
    if not save_mode:
        print("\n[1] Running inference for all horizons...")
    
    for horizon in horizons:
        if not save_mode:
            print(f"\n    --- Horizon: {horizon} hours ---")
        
        # Update prediction length
        args_config.pred_len = horizon
        
        # Load test data for this horizon
        test_dataset, test_loader = data_provider(args_config, flag='test')
        for batch_x, batch_y, _, _ in test_loader:
            demo_input = batch_x
            demo_target = batch_y
            break
        
        # Save ground truth for each horizon
        gt_np = demo_target[:, -horizon:, f_dim:].numpy()
        np.save(os.path.join(base_save_dir, f'ground_truth_t{horizon}.npy'), gt_np)
        
        # Load model
        model = DualScalePatchTST(args_config).cuda()
        checkpoint_path = checkpoint_paths[horizon]
        model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
        model.eval()
        
        # Run inference
        start = time.time()
        with torch.no_grad():
            pred = model(demo_input.float().cuda())
        infer_time = (time.time() - start) * 1000
        
        # Take only prediction length
        pred_np = pred[:, -horizon:, f_dim:].cpu().numpy()
        target_np = demo_target[:, -horizon:, f_dim:].numpy()
        
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        
        results.append((horizon, mse, mae, infer_time))
        
        # Save predictions
        np.save(os.path.join(model_save_dir, f't{horizon}_pred.npy'), pred_np)
        
        # Save metrics
        metrics = np.array([[mse, mae]])
        np.savetxt(os.path.join(model_save_dir, f't{horizon}_metrics.csv'), metrics, 
                   header='mse,mae', delimiter=',', comments='')
        
        if not save_mode:
            print(f"        MSE: {mse:.6f}, MAE: {mae:.6f}, Time: {infer_time:.2f}ms")
            print(f"        Saved to: {model_save_dir}/t{horizon}_pred.npy")
    
    # Print summary table
    if not save_mode:
        print("SUMMARY - Supervised Dual-Scale (Our Novelty)")
        print(f"{'Horizon':<10} {'MSE':<12} {'MAE':<12} {'Time(ms)':<10}")
        print("-"*44)
        for h, mse, mae, t in results:
            print(f"{h:<10} {mse:.6f}   {mae:.6f}   {t:.2f}")
        print(f"All predictions saved to: {model_save_dir}")
    else:
        # Print minimal output for orchestrator
        for h, mse, mae, t in results:
            print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, Horizon: {h}")

if __name__ == "__main__":
    run_demo()