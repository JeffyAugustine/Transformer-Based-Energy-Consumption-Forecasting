# Transformer-Based Energy Consumption Forecasting

## Project Overview

This project implements and extends PatchTST (A Time Series is Worth 64 Words) for electricity load forecasting using the ETTh1 dataset. We reproduce the supervised baseline, implement self-supervised pre-training (single and dual-scale), and propose a novel dual-scale patch encoding extension that captures both hourly fluctuations (P=8) and daily cycles (P=32) through parallel patch streams. Our supervised dual-scale model achieves competitive performance (MSE 0.377) while self-supervised pre-training for dual-scale reveals insights about task complexity and overfitting.

### Key Results:

- Supervised Dual-Scale (Our Novelty): MSE 0.377 at 96-hour forecast
- Supervised Baseline (Reproduced): MSE 0.367 at 96-hour forecast
- Self-Supervised Dual-Scale: MSE 0.878 (failure case - valuable insight)
- Best at long horizon (720h): Supervised Dual-Scale MSE 0.364

---

## Repository Structure

```text
├── PatchTST/PatchTST_supervised/           # Supervised baseline (code + results)
├── PatchTST/PatchTST_self_supervised/      # Self-supervised single-scale (code + results)
├── PatchTST/PatchTST_dual_scale/           # Supervised dual-scale - OUR NOVELTY (code + results)
├── PatchTST/PatchTST_self_supervised_dual/ # Self-supervised dual-scale - OUR NOVELTY (code + results)
│
├── notebooks/                              # Jupyter notebooks for visualization and analysis
├── demo_predictions/                       # Pre-computed predictions for all models (4 horizons × 4 models)
├── results/                                # All experiment results (loss curves, metrics CSVs)
│
├── orchestrator.py                         # Run all model demos with --save_only flag
├── demo_visualization.ipynb                # Live demo notebook with per-horizon plots
└── README.md                               # This file
```

# Reproduction Steps for Graders (Full Evaluation)

Note: Graders should follow these steps to verify all results, including trained models and experiment artifacts.

## 1. Access Zaratan HPC

```bash
ssh your_username@login.zaratan.umd.edu  # please change username
```

## 2. Clone the repository

```bash
git clone https://github.com/JeffyAugustine/Transformer-Based-Energy-Consumption-Forecasting.git
cd Transformer-Based-Energy-Consumption-Forecasting
```

## 3. Set up the environment

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

## 4. Download the ETTh1 dataset

```bash
mkdir -p data/raw
cd data/raw
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
cd ../..
```

## 5. Create dataset symlinks for each model variant

```bash
# For supervised single-scale
cd PatchTST/PatchTST_supervised
ln -s ../../data/raw dataset
cd ../..

# For self-supervised single-scale
cd PatchTST/PatchTST_self_supervised
ln -s ../../data/raw dataset
cd ../..

# For supervised dual-scale
cd PatchTST/PatchTST_dual_scale
ln -s ../../data/raw dataset
cd ../..

# For self-supervised dual-scale
cd PatchTST/PatchTST_self_supervised_dual
ln -s ../../data/raw dataset
cd ../..
```

## 6. Request a GPU and run experiments

Note: The full A100 queue has long wait times (hundreds of jobs, 1+ week). The fractional A100 slice (5GB MIG) is recommended for timely experimentation.

### Request GPU node:

```bash
srun --partition=gpu --gres=gpu:a100_1g.5gb:1 --time=04:00:00 --mem=16GB --cpus-per-task=4 --pty bash
```

### Activate environment on GPU node:

```bash
cd /path/to/Transformer-Based-Energy-Consumption-Forecasting
source env/dlpals/bin/activate
```

## 7. Train Models

### Supervised Baseline (5 runs, L=336, batch=256)

```bash
cd PatchTST/PatchTST_supervised
python run_longExp_with_logging.py \
    --is_training 1 \
    --model_id ETTh1_336_96_baseline_reproduce \
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
    --experiment_name "Baseline_d128_h8_ff256_reproduce"
```

### Self-Supervised Single-Scale (Pre-train + Fine-tune)

```bash
cd ../PatchTST_self_supervised
python run_self_supervised_with_logging.py
```

### Supervised Dual-Scale Novelty (5 runs, batch=64)

```bash
cd ../PatchTST_dual_scale
python run_dual_scale.py \
    --is_training 1 \
    --model_id ETTh1_336_96_dual_reproduce \
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
    --experiment_name "DualScale_5runs_reproduce"
```

### Self-Supervised Dual-Scale (Pre-train + Fine-tune)

```bash
cd ../PatchTST_self_supervised_dual
python run_self_supervised_dual_with_logging.py
```

Note: The run_self_supervised_dual_with_logging.py script will automatically use unique model IDs (e.g., MODEL_ID=3,4,5) to avoid overwriting existing trained models.

## 8. Testing and Demo (Using Pre-trained Models)

Note: All demo scripts and notebooks use absolute paths. If you don't want to go through path changes, the demo_visualization.ipynb notebook is pre-run with all visualizations already generated, you can simply browse through the notebook to see the results.

### Option A: Browse Pre-run Notebook (No Path Changes Needed)

Open demo_visualization.ipynb in any Jupyter environment or GitHub viewer. All visualizations, tables, and analysis are already pre-computed and saved in the notebook.

### Option B: 
### Run the Complete Demo (Requires Path Updates)

If you want to re-run the models and regenerate predictions locally, please update the absolute paths in the scripts:

```bash
cd /path/to/Transformer-Based-Energy-Consumption-Forecasting
python orchestrator.py
```

### Launch Interactive Notebook (Requires Path Updates)

```bash
jupyter notebook demo_visualization.ipynb
```

The notebook will display:

- Historical data visualization
- Per-horizon forecast plots (96, 192, 336, 720 hours)
- MSE/MAE comparison tables
- Best model identification
- Self-supervised dual-scale failure analysis

### Run Individual Model Demos (Requires Path Updates)

Each model folder contains its own demo script:

```bash
# Supervised baseline
cd PatchTST/PatchTST_supervised
python demo_supervised_single.py

# Self-supervised single-scale
cd ../PatchTST_self_supervised
python demo_self_single.py

# Supervised dual-scale (Our novelty)
cd ../PatchTST_dual_scale
python demo_supervised_dual.py

# Self-supervised dual-scale
cd ../PatchTST_self_supervised_dual
python demo_self_dual.py
```

Path Update Required: If you encounter path errors, open the respective demo script and replace the base paths (e.g., /scratch/zt1/project/msml612/user/jeffy22/...) with your own working directory path. The same applies to orchestrator.py.

## 9. Additional Visualizations and Experiment Artifacts

The notebooks/ folder contains intermediate visualization scripts we used for generating plots for the report and presentation slides. These include loss curves, ablation studies, and comparative performance charts across different model configurations.

All trained models, checkpoints, loss curves, and result CSVs are preserved within each model's respective folder (PatchTST_supervised/, PatchTST_self_supervised/, PatchTST_dual_scale/, PatchTST_self_supervised_dual/). Inside these directories, you will find:

- checkpoints/ – Individual model weights for each run
- results/ – Loss curves, prediction outputs, and evaluation metrics (MSE, MAE, RSE)
- saved_models/ – Fine-tuned and pre-trained model artifacts for self-supervised variants

These artifacts represent the many runs we performed for:

- Hyperparameter tuning (learning rate, batch size, d_model, n_heads, d_ff)
- Horizon experiments (T=96, 192, 336, 720)
- Dual-scale patch size tuning (fine: 6,8,10 / coarse: 28,32,36)
- Self-supervised mask ratio ablation (0.3, 0.4, 0.5)
- Debugging shape mismatches, RevIN denormalization, and callback coordination

All intermediate results are kept for transparency and reproducibility. The final reported numbers in the paper and demo are derived from the best configurations found during this process.

# For Non-Graders / Lightweight Installation

If you are not a grader and simply want to explore the code without downloading all intermediate experiment files (checkpoints, results, predictions, demo artifacts), clone the repository and clean locally.

## Recommended: Clone and Clean Local Files

```bash
git clone https://github.com/JeffyAugustine/Transformer-Based-Energy-Consumption-Forecasting.git
cd Transformer-Based-Energy-Consumption-Forecasting

# Remove large experiment artifacts (keep code only)
rm -rf PatchTST/*/checkpoints
rm -rf PatchTST/*/results
rm -rf PatchTST/*/saved_models
rm -rf demo_predictions
rm -rf results
rm -f orchestrator.py
rm -f demo_visualization.ipynb
```

This leaves only the source code. You can still train models from scratch using the same training commands specified in Section 7 above.

## Updated .gitignore for Your Repository

To prevent these files from being re-uploaded in future commits, add or update the .gitignore file in your repository root:

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
env/
dlpals/
.venv/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints/

# Experiment outputs (checkpoints, results, predictions)
**/checkpoints/
**/results/
**/saved_models/
demo_predictions/
results/
*.npy
*.csv
*.pth
*.log

# Demo and visualization (optional: exclude from lightweight clone)
orchestrator.py
demo_visualization.ipynb
notebooks/

# OS metadata
.DS_Store
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
```

# Citation

```bibtex
@inproceedings{Nie2023PatchTST,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
