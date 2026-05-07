import subprocess
import os
import sys
import time
from datetime import datetime

# Base directory
BASE_DIR = '/scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting'

# Model configurations
models = [
    {
        'name': 'Supervised Single',
        'folder': 'PatchTST/PatchTST_supervised',
        'script': 'demo_supervised_single.py',
        'horizons': [96, 192, 336, 720]
    },
    {
        'name': 'Supervised Dual',
        'folder': 'PatchTST/PatchTST_dual_scale',
        'script': 'demo_supervised_dual.py',
        'horizons': [96, 192, 336, 720]
    },
    {
        'name': 'Self-Supervised Single',
        'folder': 'PatchTST/PatchTST_self_supervised',
        'script': 'demo_self_single.py',
        'horizons': [96, 192, 336, 720]
    },
    {
        'name': 'Self-Supervised Dual',
        'folder': 'PatchTST/PatchTST_self_supervised_dual',
        'script': 'demo_self_dual.py',
        'horizons': [96]
    }
]

def run_demo(model_info):
    """Run a single demo script and return its output"""
    print(f" Running: {model_info['name']}")
    
    folder_path = os.path.join(BASE_DIR, model_info['folder'])
    script_path = os.path.join(folder_path, model_info['script'])
    
    if not os.path.exists(script_path):
        print(f"    Script not found: {script_path}")
        return None
    
    # Change to the folder and run the script
    cmd = f"cd {folder_path} && python {model_info['script']} --save_only"
    
    print(f"    Script: {model_info['script']}")
    print(f"    Running...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"    Completed in {elapsed:.2f} seconds")
            # Parse output to extract MSE/MAE for each horizon
            output_lines = result.stdout.strip().split('\n')
            metrics = {}
            for line in output_lines:
                if line.startswith('MSE:'):
                    parts = line.split(',')
                    mse = float(parts[0].split(':')[1].strip())
                    mae = float(parts[1].split(':')[1].strip())
                    horizon = int(parts[2].split(':')[1].strip())
                    metrics[horizon] = {'mse': mse, 'mae': mae}
            return metrics
        else:
            print(f"    Failed with error: {result.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"    Timeout after 600 seconds")
        return None
    except Exception as e:
        print(f"    Error: {str(e)}")
        return None

def print_summary(all_results):
    """Print a summary table of all results"""
    print(" FINAL SUMMARY - ALL MODELS")
    
    # Header
    print(f"\n{'Model':<28}", end='')
    for horizon in [96, 192, 336, 720]:
        print(f"  T={horizon:<18}", end='')
    print()
    
    # Results for each model
    for model_name, metrics in all_results.items():
        print(f"{model_name:<28}", end='')
        for horizon in [96, 192, 336, 720]:
            if horizon in metrics:
                mse = metrics[horizon]['mse']
                mae = metrics[horizon]['mae']
                print(f"  MSE:{mse:.4f} MAE:{mae:.4f}  ", end='')
            else:
                print(f"  {'MISSING':<20}", end='')
        print()
    
    
    # Highlight best model for each horizon
    print("\n BEST PERFORMANCE BY HORIZON:")
    
    for horizon in [96, 192, 336, 720]:
        best_model = None
        best_mse = float('inf')
        for model_name, metrics in all_results.items():
            if horizon in metrics and metrics[horizon]['mse'] < best_mse:
                best_mse = metrics[horizon]['mse']
                best_model = model_name
        if best_model:
            print(f"   T={horizon:<4} → {best_model:<28} (MSE: {best_mse:.6f})")
        else:
            print(f"   T={horizon:<4} → No data available")
    

def check_predictions_directory():
    """Check what predictions have been saved"""
    predictions_dir = os.path.join(BASE_DIR, 'demo_predictions')
    if os.path.exists(predictions_dir):
        print("\n Predictions saved to:")
        print(f"   {predictions_dir}")
        
        # Show what's inside
        for folder in ['supervised_single', 'supervised_dual', 'self_single', 'self_dual']:
            folder_path = os.path.join(predictions_dir, folder)
            if os.path.exists(folder_path):
                files = os.listdir(folder_path)
                pred_files = [f for f in files if f.endswith('.npy')]
                print(f"    {folder}: {len(pred_files)} prediction files")
    else:
        print("\n Predictions directory not found")

def main():
    print(" MSML612 - Model Demo Orchestrator")
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Base directory: {BASE_DIR}")
    
    start_total = time.time()
    
    all_results = {}
    
    for model in models:
        metrics = run_demo(model)
        if metrics:
            all_results[model['name']] = metrics
        else:
            all_results[model['name']] = {}
    
    # Print final summary
    print_summary(all_results)
    
    # Check predictions directory
    check_predictions_directory()
    
    total_time = time.time() - start_total
    print(f"\n Total execution time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()