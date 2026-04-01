import os
import sys
import time
import json
import subprocess
import pandas as pd
from datetime import datetime
import torch

def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,power.limit', '--format=csv,noheader'], 
                                capture_output=True, text=True)
        gpu_info = result.stdout.strip().split(',')
        return {
            'gpu_name': gpu_info[0].strip(),
            'gpu_memory': gpu_info[1].strip(),
            'gpu_power': gpu_info[2].strip() if len(gpu_info) > 2 else 'N/A'
        }
    except:
        return {'gpu_name': 'Unknown', 'gpu_memory': 'Unknown', 'gpu_power': 'Unknown'}

def log_experiment(config, results, log_file="experiment_log.csv"):
    """Log experiment results to CSV file"""
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    # Create record
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_id': config.get('model_id', 'unknown'),
        'dataset': config.get('data', 'unknown'),
        'seq_len': config.get('seq_len', 336),
        'pred_len': config.get('pred_len', 96),
        'd_model': config.get('d_model', 128),
        'n_heads': config.get('n_heads', 8),
        'e_layers': config.get('e_layers', 3),
        'd_ff': config.get('d_ff', 256),
        'batch_size': config.get('batch_size', 256),
        'learning_rate': config.get('learning_rate', 0.0001),
        'train_epochs': config.get('train_epochs', 100),
        'patience': config.get('patience', 10),
        'itr': config.get('itr', 5),
        'gpu_name': gpu_info['gpu_name'],
        'gpu_memory': gpu_info['gpu_memory'],
        'mse': results.get('mse', None),
        'mae': results.get('mae', None),
        'best_epoch': results.get('best_epoch', None),
        'total_time_seconds': results.get('total_time', None)
    }
    
    # Check if log file exists
    log_path = os.path.join(os.path.dirname(__file__), 'experiment_logs.csv')
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    
    df.to_csv(log_path, index=False)
    print(f"✅ Experiment logged to {log_path}")
    return df

if __name__ == "__main__":
    print("This is a helper module. Import log_experiment in your training script.")
