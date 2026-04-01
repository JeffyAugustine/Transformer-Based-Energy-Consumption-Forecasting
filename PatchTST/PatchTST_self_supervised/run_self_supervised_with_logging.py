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
    csv_path = os.path.join(results_dir, 'self_supervised_results.csv')
    df_new = pd.DataFrame([results])
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)
    print(f" Results saved to: {csv_path}")
    return csv_path

if __name__ == "__main__":
    start_time = time.time()
    start_datetime = datetime.now()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    print("SELF-SUPERVISED PATCHTST - ETTh1")
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']})")
    
    print("\n STAGE 1: Pre-training (Masked Autoencoder)")
    
    pretrain_start = time.time()
    
    pretrain_cmd = [
        "python", "patchtst_pretrain.py",
        "--dset_pretrain", "etth1",
        "--context_points", "512",
        "--patch_len", "12",
        "--stride", "12",
        "--mask_ratio", "0.4",
        "--batch_size", "64",
        "--n_epochs_pretrain", "100",
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
        "--pretrained_model_id", "1"
    ]
    
    print("Running pre-training...")
    # Run without capture to see live output
    process = subprocess.Popen(pretrain_cmd, text=True)
    process.wait()
    
    if process.returncode != 0:
        print(" Pre-training failed!")
        exit(1)
    
    pretrain_time = time.time() - pretrain_start
    print(f" Pre-training completed in {pretrain_time/60:.2f} minutes")
    
    model_path = "saved_models/etth1/masked_patchtst/based_model/"
    model_files = glob.glob(model_path + "*.pth")
    if model_files:
        pretrained_model = model_files[0]
        print(f" Model saved to: {pretrained_model}.pth")
    else:
        print(" No model file found! Using expected path.")
        pretrained_model = model_path + "patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain100_mask0.4_model1"
    
    print("\n STAGE 2: Fine-tuning for Forecasting")
    
    finetune_start = time.time()
    
    finetune_cmd = [
        "python", "patchtst_finetune.py",
        "--is_finetune", "1",
        "--dset_finetune", "etth1",
        "--pretrained_model", pretrained_model,
        "--context_points", "336",
        "--target_points", "96",
        "--patch_len", "16",
        "--stride", "8",
        "--batch_size", "128",
        "--n_epochs_finetune", "20",
        "--lr", "1e-4",
        "--n_layers", "3",
        "--n_heads", "4",
        "--d_model", "16",
        "--d_ff", "128",
        "--dropout", "0.2",
        "--head_dropout", "0.2",
        "--revin", "1",
        "--features", "M",
        "--num_workers", "4",
        "--finetuned_model_id", "1"
    ]
    
    print("Running fine-tuning...")
    # Run without capture to see live output
    process = subprocess.Popen(finetune_cmd, text=True)
    process.wait()
    
    if process.returncode != 0:
        print(" Fine-tuning failed!")
        exit(1)
    
    finetune_time = time.time() - finetune_start
    print(f" Fine-tuning completed in {finetune_time/60:.2f} minutes")
    
    total_time = time.time() - start_time
    
    # Save results to CSV (without parsing since we saw live output)
    results = {
        'timestamp': start_datetime.isoformat(),
        'experiment_name': 'Self_Supervised_ETTh1_336_96',
        'dataset': 'ETTh1',
        'seq_len': 336,
        'pred_len': 96,
        'pretrain_epochs': 100,
        'finetune_epochs': 20,
        'mask_ratio': 0.4,
        'pretrain_patch_len': 12,
        'pretrain_stride': 12,
        'finetune_patch_len': 16,
        'finetune_stride': 8,
        'gpu_name': gpu_info['gpu_name'],
        'gpu_memory': gpu_info['gpu_memory'],
        'pretrain_time_min': pretrain_time/60,
        'finetune_time_min': finetune_time/60,
        'total_time_min': total_time/60,
        'mse': None,
        'mae': None,
        'notes': 'Self-supervised with masked autoencoder, fine-tuned on forecasting. Check terminal output for MSE/MAE.'
    }
    
    results_dir = "/scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting/results/jeffy22"
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(results, results_dir)
    
    print("SELF-SUPERVISED TRAINING COMPLETE!")
    print(f"Total time: {total_time/60:.2f} minutes")