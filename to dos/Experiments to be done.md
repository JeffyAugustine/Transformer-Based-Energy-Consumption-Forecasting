# Dual-Scale PatchTST - Experiments to do 

Hey guys, please spend some time running these commands. When you're done, edit this README and mark what's completed (✅) and what's not (❌). Add any observations at the bottom.

**Important:** All scripts use logging and results will be appended to existing CSV files, not overwritten. Each experiment has a unique `--experiment_name` so we can track everything separately.

---

## Setup (Everytime you login to HPC)

### Request GPU (Please use the same command so we use the same hardware throughout)

```bash
srun --partition=gpu --gres=gpu:a100_1g.5gb:1 --time=04:00:00 --mem=16GB --cpus-per-task=4 --pty 
```
### Activate Environment
```bash
cd /scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting
source env/dlpals/bin/activate
```
## Section 1: All Horizons on All Models (Most Important)
Run T=96, 192, 336, 720 for each model. This tells us which horizon benefits most from dual-scale. Complete this section before moving to Section 2.

### 1.1 Supervised Baseline (Single Scale)
```bash
cd PatchTST/PatchTST_supervised
```
### Horizon	Command
- T=96
```bash
python run_longExp_with_logging.py --is_training 1 --model_id ETTh1_336_96_baseline --model PatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 96 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --batch_size 256 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "Baseline_T96"
   ```
- T=192	
```bash
python run_longExp_with_logging.py --is_training 1 --model_id ETTh1_336_192_baseline --model PatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 192 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --batch_size 256 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "Baseline_T192"
```
- T=336	
```bash
python run_longExp_with_logging.py --is_training 1 --model_id ETTh1_336_336_baseline --model PatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 336 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --batch_size 256 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "Baseline_T336"
```
- T=720	
```bash
python run_longExp_with_logging.py --is_training 1 --model_id ETTh1_336_720_baseline --model PatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 720 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --batch_size 256 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "Baseline_T720"
```
Status: T=96 ✅ | T=192 ❌ | T=336 ❌ | T=720 ❌

### 1.2 Self-Supervised (Single Scale) 
```bash
cd ../PatchTST_self_supervised
```
First, edit the script to change the horizon. Open run_self_supervised_with_logging.py and find the ```--target_points``` argument (line 104). Change it to the desired horizon. Also change experiment name. Change ```'experiment_name': 'Self_Supervised_ETTh1_336_96'``` in line 139.

### For T=96 (already done)
```bash
python run_self_supervised_with_logging.py
```
For other horizons, please do changes and run above command.

Status: T=96 ✅ | T=192 ❌ | T=336 ❌ | T=720 ❌

### 1.3 Dual-Scale
```bash
cd ../PatchTST_dual_scale
```
### Horizon	Command
- T=96	
```bash
python run_dual_scale.py --is_training 1 --model_id ETTh1_336_96_dual --model DualScalePatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 96 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --patch_len_fine 8 --stride_fine 4 --patch_len_coarse 32 --stride_coarse 16 --batch_size 64 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "DualScale_T96"
```
- T=192	
```bash
python run_dual_scale.py --is_training 1 --model_id ETTh1_336_192_dual --model DualScalePatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 192 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --patch_len_fine 8 --stride_fine 4 --patch_len_coarse 32 --stride_coarse 16 --batch_size 64 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "DualScale_T192"
```
- T=336	
```bash
python run_dual_scale.py --is_training 1 --model_id ETTh1_336_336_dual --model DualScalePatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 336 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --patch_len_fine 8 --stride_fine 4 --patch_len_coarse 32 --stride_coarse 16 --batch_size 64 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "DualScale_T336"
```
- T=720	
```bash
python run_dual_scale.py --is_training 1 --model_id ETTh1_336_720_dual --model DualScalePatchTST --data ETTh1 --features M --root_path ./dataset/ --data_path ETTh1.csv --seq_len 336 --pred_len 720 --e_layers 3 --d_model 128 --n_heads 8 --d_ff 256 --patch_len_fine 8 --stride_fine 4 --patch_len_coarse 32 --stride_coarse 16 --batch_size 64 --train_epochs 100 --patience 10 --itr 5 --use_gpu True --experiment_name "DualScale_T720"
```
Status: T=96 ✅ | T=192 ❌ | T=336 ❌ | T=720 ❌

## Section 2: Hyperparameter Tuning for Dual-Scale
First, complete Section 1 to understand which horizon gives the best opportunity for improvement. Then run these tuning experiments at that horizon.

**Important**: As you run these experiments, you'll get a better sense of what works and what doesn't. Feel free to add or modify experiments based on what you observe. Just make sure to change the --experiment_name to something descriptive so nothing gets overwritten. I'll blindly trust you guys on this.

### 2.1 Patch Size Combinations
| Experiment | Fine P | Coarse P | Command Additions |
|------------|--------|----------|-------------------|
| Current (baseline) | 8 | 32 | Already done |
| P12_P24 | 12 | 24 | `--patch_len_fine 12 --stride_fine 6 --patch_len_coarse 24 --stride_coarse 12` |
| P6_P48 | 6 | 48 | `--patch_len_fine 6 --stride_fine 3 --patch_len_coarse 48 --stride_coarse 24` |
| P16_P32 | 16 | 32 | `--patch_len_fine 16 --stride_fine 8 --patch_len_coarse 32 --stride_coarse 16` |
| P10_P40 | 10 | 40 | `--patch_len_fine 10 --stride_fine 5 --patch_len_coarse 40 --stride_coarse 20` |

Full command template (replace [ARGS] with above and [NAME] in the last line):

```bash
cd ../PatchTST_dual_scale

python run_dual_scale.py \
    --is_training 1 \
    --model_id ETTh1_336_96_dual_patch_tune \
    --model DualScalePatchTST \
    --data ETTh1 \
    --features M \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --seq_len 336 \
    --pred_len 96 \
    --e_layers 3 \
    --d_model 128 \
    --n_heads 8 \
    --d_ff 256 \
    --batch_size 64 \
    --train_epochs 100 \
    --patience 10 \
    --itr 3 \
    --use_gpu True \
    --experiment_name "DualScale_PatchTune_[NAME]" \
    [ADD PATCH ARGS HERE]
```

Status: P8_P32 ✅ | P12_P24 ❌ | P6_P48 ❌ | P16_P32 ❌ | P10_P40 ❌

### 2.2 Learning Rate
| Experiment | Learning Rate | Command Addition |
|------------|---------------|------------------|
| Current | 1e-4 | Already done |
| LR_5e-5 | 5e-5 | `--learning_rate 0.00005` |
| LR_2e-4 | 2e-4 | `--learning_rate 0.0002` |
| LR_1e-3 | 1e-3 | `--learning_rate 0.001` |

Full command template (replace [ARGS] with above and [NAME] in the last line):

```bash
python run_dual_scale.py \
    --is_training 1 \
    --model_id ETTh1_336_96_dual_lr_tune \
    --model DualScalePatchTST \
    --data ETTh1 \
    --features M \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --seq_len 336 \
    --pred_len 96 \
    --e_layers 3 \
    --d_model 128 \
    --n_heads 8 \
    --d_ff 256 \
    --patch_len_fine 8 \
    --stride_fine 4 \
    --patch_len_coarse 32 \
    --stride_coarse 16 \
    --batch_size 64 \
    --train_epochs 100 \
    --patience 10 \
    --itr 3 \
    --use_gpu True \
    --experiment_name "DualScale_LR_Tune_[NAME]" \
    [ADD LR ARGS HERE]
```
Status: LR_1e-4 ✅ | LR_5e-5 ❌ | LR_2e-4 ❌ | LR_1e-3 ❌

### 2.3 Model Capacity (d_model, n_heads, d_ff, dropout)
| Experiment | d_model | n_heads | d_ff | dropout |
|------------|---------|---------|------|---------|
| Current | 128 | 8 | 256 | 0.05 |
| Smaller | 64 | 4 | 128 | 0.1 |
| Larger | 256 | 16 | 512 | 0.05 |

Full command template:

```bash
python run_dual_scale.py \
    --is_training 1 \
    --model_id ETTh1_336_96_dual_capacity_tune \
    --model DualScalePatchTST \
    --data ETTh1 \
    --features M \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --seq_len 336 \
    --pred_len 96 \
    --e_layers 3 \
    --patch_len_fine 8 \
    --stride_fine 4 \
    --patch_len_coarse 32 \
    --stride_coarse 16 \
    --batch_size 64 \
    --train_epochs 100 \
    --patience 10 \
    --itr 3 \
    --use_gpu True \
    --experiment_name "DualScale_Capacity_Tune_[NAME]" \
    [ADD CAPACITY ARGS HERE]
```

Status: Default ✅ | Smaller ❌ | Larger ❌

### 2.4 Stride Overlap
| Experiment | Fine Stride | Coarse Stride | Patches | Command Addition |
|------------|-------------|---------------|---------|------------------|
| Current | 4 | 16 | 105 | Already done |
| Less Overlap | 6 | 24 | ~70 | `--stride_fine 6 --stride_coarse 24` |
| More Overlap | 2 | 8 | ~161 | `--stride_fine 2 --stride_coarse 8` |

Status: Current ✅ | Less Overlap ❌ | More Overlap ❌

## Section 3: Observations (Please Fill after each run)
| Experiment | Observation | Who Ran |
|------------|-------------|---------|
| Example: DualScale_T192 | Converged at epoch 18, MSE 0.385 | Jeffy |
| | | |
| | | |
| | | |

## Section 4: Results Location
All results are automatically saved to:
```
/scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting/results/jeffy22/

experiment_results.csv - Supervised experiments

self_supervised_results.csv - Self-supervised experiments

dual_scale_results.csv - Dual-scale experiments

Loss curves are saved in each model's ./results/ folder.
```
Notes
Don't worry about overwriting — logging appends to CSV, doesn't replace as long as you guys give different names so dont forget to change name.

Each run takes ~8-15 minutes depending on horizon.

Batch size is fixed at 64 due to MIG slice memory limit (5GB)

Feel free to add experiments — as you run these, you'll get a better sense of what to try next. Just change --experiment_name to something descriptive so we don't lose any results

## Less goo!!!
