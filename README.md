# Transformer-Based Energy Consumption Forecasting

## Project Overview

This project implements and extends PatchTST (A Time Series is Worth 64 Words) for electricity load forecasting using the ETTh1 dataset. We reproduce the supervised baseline, evaluate paper hyperparameters, implement self-supervised pre-training, and propose a novel **dual-scale patch encoding extension** that captures both hourly fluctuations (P=8) and daily cycles (P=32) through parallel patch streams.

---

## Repository Structure

```
├── PatchTST/PatchTST_supervised/ # Supervised baseline and paper hyperparams
├── PatchTST/PatchTST_self_supervised/ # Self-supervised pre-training + fine-tuning
├── PatchTST/PatchTST_dual_scale/ # Dual-scale novelty implementation
├── notebooks/ # Jupyter notebooks for visualizations
└── README.md # This file
```

---

## Reproduction Steps

### 1. Access Zaratan HPC

```bash
ssh your_username@login.zaratan.umd.edu # please change username
```

### 2. Clone the repository

```bash
git clone https://github.com/JeffyAugustine/Transformer-Based-Energy-Consumption-Forecasting.git
cd Transformer-Based-Energy-Consumption-Forecasting
```

### 3. Set up the environment

```bash
# Load Python module
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2

# Create virtual environment
python -m venv env/dlpals
source env/dlpals/bin/activate

# Install dependencies
pip install torch==1.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scikit-learn tqdm einops typing_extensions accelerate
```

### 4. Download the ETTh1 dataset

```bash
mkdir -p data/raw
cd data/raw
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
cd ../..
```

### 5. Create dataset symlinks for each model variant

```bash
# For supervised
cd PatchTST/PatchTST_supervised
ln -s ../../data/raw dataset
cd ../..

# For self-supervised
cd PatchTST/PatchTST_self_supervised
ln -s ../../data/raw dataset
cd ../..

# For dual-scale
cd PatchTST/PatchTST_dual_scale
ln -s ../../data/raw dataset
cd ../..
```

### 6. Request a GPU and run experiments

*Note: The full A100 queue has long wait times (hundreds of jobs, 1+ week). The fractional A100 slice is recommended for timely experimentation.*

#### Request GPU node:

```bash
srun --partition=gpu --gres=gpu:a100_1g.5gb:1 --time=04:00:00 --mem=16GB --cpus-per-task=4 --pty bash
```

#### Activate environment on GPU node:

```bash
cd /path/to/Transformer-Based-Energy-Consumption-Forecasting
source env/dlpals/bin/activate
```

#### Supervised Baseline (5 runs, L=336, batch=256)

```bash
cd PatchTST/PatchTST_supervised
python run_longExp_with_logging.py \
    --is_training 1 \
    --model_id ETTh1_336_96_baseline \
    --model PatchTST \
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
    --batch_size 256 \
    --train_epochs 100 \
    --patience 10 \
    --itr 5 \
    --use_gpu True \
    --experiment_name "Baseline_d128_h8_ff256"
```

#### Paper Hyperparameters (5 runs, d_model=16)

```bash
python run_longExp_with_logging.py \
    --is_training 1 \
    --model_id ETTh1_336_96_paper \
    --model PatchTST \
    --data ETTh1 \
    --features M \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --seq_len 336 \
    --pred_len 96 \
    --e_layers 3 \
    --d_model 16 \
    --n_heads 4 \
    --d_ff 128 \
    --dropout 0.2 \
    --batch_size 256 \
    --train_epochs 100 \
    --patience 10 \
    --itr 5 \
    --use_gpu True \
    --experiment_name "Paper_Hyperparams_d16_h4_ff128"
```

#### Self-Supervised (Pre-train + Fine-tune with logging)

```bash
cd ../PatchTST_self_supervised
python run_self_supervised_with_logging.py
```

#### Dual-Scale Novelty (5 runs, batch=64)

```bash
cd ../PatchTST_dual_scale
python run_dual_scale.py \
    --is_training 1 \
    --model_id ETTh1_336_96_dual \
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
    --itr 5 \
    --use_gpu True \
    --experiment_name "DualScale_5runs"
```

### 7. Visuals and results notebook

Our notebooks in the notebook folder has code for generating visuals and tables.

---

## Citation

```bibtex
@inproceedings{Nie2023PatchTST,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
