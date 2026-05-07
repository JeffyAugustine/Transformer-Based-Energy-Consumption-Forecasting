import subprocess
import time
import json
import os
import re
import pandas as pd
import glob
from datetime import datetime

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                '--format=csv,noheader'], capture_output=True, text=True)
        gpu_info = result.stdout.strip().split(',')
        return {
            'gpu_name': gpu_info[0].strip(),
            'gpu_memory': gpu_info[1].strip(),
            'cuda_version': gpu_info[2].strip() if len(gpu_info) > 2 else 'Unknown'
        }
    except:
        return {'gpu_name': 'Unknown', 'gpu_memory': 'Unknown', 'cuda_version': 'Unknown'}

def save_results(results, results_dir):
    csv_path = os.path.join(results_dir, 'self_supervised_dual_results.csv')
    df_new = pd.DataFrame([results])
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)
    print(f" Results saved to: {csv_path}")
    return csv_path

def get_model_signature(args_dict):
    """Generate a unique signature for the model based on hyperparameters"""
    key_params = [
        'context_points', 'patch_len_fine', 'stride_fine', 
        'patch_len_coarse', 'stride_coarse', 'mask_ratio',
        'd_model', 'n_heads', 'n_layers', 'd_ff'
    ]
    signature_parts = [f"{k}={args_dict[k]}" for k in key_params if k in args_dict]
    return "_".join(signature_parts)

if __name__ == "__main__":
    start_time = time.time()
    start_datetime = datetime.now()
    
    gpu_info = get_gpu_info()
    
    print("DUAL-SCALE SELF-SUPERVISED PATCHTST - ETTh1")
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']})")
    
    MODEL_ID = 4  
    
    pretrain_config = {
        'dset_pretrain': 'etth1',
        'context_points': 512,
        'patch_len_fine': 8,
        'stride_fine': 4,
        'patch_len_coarse': 32,
        'stride_coarse': 16,
        'mask_ratio': 0.4,
        'batch_size': 32,
        'n_epochs_pretrain': 100,
        'lr': 1e-4,
        'n_layers': 3,
        'n_heads': 16,
        'd_model': 128,
        'd_ff': 512,
        'dropout': 0.2,
        'head_dropout': 0.2,
        'revin': 1,
        'features': 'M',
        'num_workers': 4,
        'pretrained_model_id': MODEL_ID
    }
    
    # Generate model signature to check if pre-trained model exists
    model_signature = f"cw{pretrain_config['context_points']}_fineP{pretrain_config['patch_len_fine']}_coarseP{pretrain_config['patch_len_coarse']}_mask{pretrain_config['mask_ratio']}_model{pretrain_config['pretrained_model_id']}"
    model_path_dir = f"saved_models/{pretrain_config['dset_pretrain']}/masked_patchtst_dual/based_model/"
    
    # ONLY match pre-trained models 
    expected_model_pattern = f"{model_path_dir}*pretrained*{model_signature}*.pth"
    existing_models = glob.glob(expected_model_pattern)
    
    print("\n Pre-training (Masked Autoencoder with Dual-Scale)")
    
    pretrain_start = time.time()
    
    if existing_models:
        pretrained_model = existing_models[0]
        print(f" Pre-trained model already exists: {pretrained_model}")
        print(" Skipping pre-training...")
        pretrain_time = 0
    else:
        print(" No existing pre-trained model found. Running pre-training...")
        
        pretrain_cmd = [
            "python", "patchtst_pretrain_dual.py",
            "--dset_pretrain", str(pretrain_config['dset_pretrain']),
            "--context_points", str(pretrain_config['context_points']),
            "--patch_len_fine", str(pretrain_config['patch_len_fine']),
            "--stride_fine", str(pretrain_config['stride_fine']),
            "--patch_len_coarse", str(pretrain_config['patch_len_coarse']),
            "--stride_coarse", str(pretrain_config['stride_coarse']),
            "--mask_ratio", str(pretrain_config['mask_ratio']),
            "--batch_size", str(pretrain_config['batch_size']),
            "--n_epochs_pretrain", str(pretrain_config['n_epochs_pretrain']),
            "--lr", str(pretrain_config['lr']),
            "--n_layers", str(pretrain_config['n_layers']),
            "--n_heads", str(pretrain_config['n_heads']),
            "--d_model", str(pretrain_config['d_model']),
            "--d_ff", str(pretrain_config['d_ff']),
            "--dropout", str(pretrain_config['dropout']),
            "--head_dropout", str(pretrain_config['head_dropout']),
            "--revin", str(pretrain_config['revin']),
            "--features", pretrain_config['features'],
            "--num_workers", str(pretrain_config['num_workers']),
            "--pretrained_model_id", str(pretrain_config['pretrained_model_id'])
        ]
        
        process = subprocess.Popen(pretrain_cmd, text=True)
        process.wait()
        
        if process.returncode != 0:
            print(" Pre-training failed!")
            exit(1)
        
        pretrain_time = time.time() - pretrain_start
        print(f" Pre-training completed in {pretrain_time/60:.2f} minutes")
        
        model_files = glob.glob(model_path_dir + "*pretrained*.pth")
        if model_files:
            pretrained_model = max(model_files, key=os.path.getctime)
        else:
            print(" No model file found!")
            exit(1)
    
    print(f" Using pre-trained model: {pretrained_model}")
    
    finetuned_pattern = f"{model_path_dir}*finetuned*_model{MODEL_ID}.pth"
    existing_finetuned = glob.glob(finetuned_pattern)
    
    print("\n STAGE 2: Fine-tuning for Forecasting")
    
    finetune_start = time.time()
    
    if existing_finetuned:
        finetuned_model = existing_finetuned[0]
        print(f" Fine-tuned model already exists: {finetuned_model}")
        print(" Skipping fine-tuning...")
        finetune_time = 0
    else:
        print(" No existing fine-tuned model found. Running fine-tuning...")
        
        finetune_cmd = [
            "python", "patchtst_finetune_dual.py",
            "--is_finetune", "1",
            "--dset_finetune", "etth1",
            "--pretrained_model", pretrained_model,
            "--context_points", "512",
            "--target_points", "96",
            "--patch_len_fine", "8",
            "--stride_fine", "4",
            "--patch_len_coarse", "32",
            "--stride_coarse", "16",
            "--batch_size", "32",
            "--n_epochs_finetune", "20",
            "--lr", "1e-4",
            "--n_layers", "3",
            "--n_heads", "16",
            "--d_model", "128",
            "--d_ff", "512",
            "--dropout", "0.2",
            "--head_dropout", "0.2",
            "--revin", "1",
            "--features", "M",
            "--num_workers", "4",
            "--finetuned_model_id", str(MODEL_ID)
        ]
        
        print("Running fine-tuning...")
        process = subprocess.Popen(finetune_cmd, text=True)
        process.wait()
        
        if process.returncode != 0:
            print(" Fine-tuning failed!")
            exit(1)
        
        finetune_time = time.time() - finetune_start
        print(f" Fine-tuning completed in {finetune_time/60:.2f} minutes")
    
    total_time = time.time() - start_time
    
    acc_pattern = f"{model_path_dir}*finetuned*_model{MODEL_ID}_acc.csv"
    acc_files = glob.glob(acc_pattern)
    
    mse = None
    mae = None
    if acc_files:
        df_acc = pd.read_csv(acc_files[-1])
        mse = df_acc['mse'].iloc[0] if 'mse' in df_acc.columns else None
        mae = df_acc['mae'].iloc[0] if 'mae' in df_acc.columns else None
        print(f" Results found: MSE={mse}, MAE={mae}")
    else:
        print(f" No results file found for model ID {MODEL_ID}")
    
    results = {
        'timestamp': start_datetime.isoformat(),
        'experiment_name': f'Self_Supervised_Dual_ETTh1_336_96_model{MODEL_ID}',
        'dataset': 'ETTh1',
        'seq_len': 336,
        'pred_len': 96,
        'model_id': MODEL_ID,
        'pretrain_epochs': pretrain_config['n_epochs_pretrain'],
        'finetune_epochs': 20,
        'mask_ratio': pretrain_config['mask_ratio'],
        'patch_len_fine': pretrain_config['patch_len_fine'],
        'stride_fine': pretrain_config['stride_fine'],
        'patch_len_coarse': pretrain_config['patch_len_coarse'],
        'stride_coarse': pretrain_config['stride_coarse'],
        'gpu_name': gpu_info['gpu_name'],
        'gpu_memory': gpu_info['gpu_memory'],
        'pretrain_time_min': pretrain_time/60 if pretrain_time > 0 else 0,
        'finetune_time_min': finetune_time/60,
        'total_time_min': total_time/60,
        'mse': mse,
        'mae': mae,
        'notes': f'Dual-Scale Self-Supervised with masked autoencoder, fine-tuned on forecasting. Model ID {MODEL_ID}'
    }
    
    results_dir = "/scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting/results/jeffy22"
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(results, results_dir)
    
    print("DUAL-SCALE SELF-SUPERVISED TRAINING COMPLETE!")
    print(f"Model ID: {MODEL_ID}")
    print(f"Total time: {total_time/60:.2f} minutes")
    if mse:
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
