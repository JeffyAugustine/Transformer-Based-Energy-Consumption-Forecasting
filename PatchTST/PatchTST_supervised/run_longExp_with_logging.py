import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import time
import json
import subprocess
from datetime import datetime
import pandas as pd

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

def save_results_to_csv(all_results, results_file='experiment_results.csv'):
    results_dir = os.path.join(os.path.dirname(__file__), '../../results/jeffy22')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, results_file)
    
    df_new = pd.DataFrame(all_results)
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)
    print(f" Results saved to: {csv_path}")
    return csv_path

if __name__ == '__main__':
    start_time = time.time()
    start_datetime = datetime.now()
    
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # PatchTST specific
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Transformer params
    parser.add_argument('--embed_type', type=int, default=0, help='embed type')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--do_predict', action='store_true', help='predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', default=False, help='mixed precision')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')
    parser.add_argument('--test_flop', action='store_true', default=False, help='test flop')

    # Logging
    parser.add_argument('--log_results', type=bool, default=True, help='log results to CSV')
    parser.add_argument('--experiment_name', type=str, default='auto', help='experiment name for logging')

    args = parser.parse_args()

    # Set seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    gpu_info = get_gpu_info()
    
    all_run_results = []
    
    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            run_start = time.time()
            
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers,
                args.d_layers, args.d_ff, args.factor,
                args.embed, args.distil, args.des, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae, rse = exp.test(setting)
            
            run_end = time.time()
            run_time = run_end - run_start
            
            print(f"Run {ii+1} completed in {run_time/60:.2f} minutes")
            
            result_entry = {
                'timestamp': start_datetime.isoformat(),
                'experiment_name': args.experiment_name if args.experiment_name != 'auto' else args.model_id,
                'run': ii,
                'model_id': args.model_id,
                'model': args.model,
                'dataset': args.data,
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'e_layers': args.e_layers,
                'd_ff': args.d_ff,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'dropout': args.dropout,
                'patch_len': args.patch_len,
                'stride': args.stride,
                'train_epochs': args.train_epochs,
                'patience': args.patience,
                'itr_total': args.itr,
                'gpu_name': gpu_info['gpu_name'],
                'gpu_memory': gpu_info['gpu_memory'],
                'cuda_version': gpu_info['cuda_version'],
                'run_time_seconds': run_time,
                'mse': mse,
                'mae': mae,
                'rse': rse,
                'notes': ''
            }
            
            all_run_results.append(result_entry)
            
            if args.do_predict:
                exp.predict(setting, True)
            
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers,
            args.d_layers, args.d_ff, args.factor,
            args.embed, args.distil, args.des, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"All experiments completed!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"GPU used: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']})")
    
    if args.log_results and all_run_results:
        csv_path = save_results_to_csv(all_run_results)
        print(f"\n Experiment results logged to: {csv_path}")
        
        json_path = os.path.join(os.path.dirname(csv_path), f"{args.model_id}_summary.json")
        summary = {
            'experiment': args.model_id,
            'dataset': args.data,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'num_runs': len(all_run_results),
            'total_time_minutes': total_time/60,
            'gpu': gpu_info,
            'hyperparameters': {
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'e_layers': args.e_layers,
                'd_ff': args.d_ff,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'dropout': args.dropout,
                'patch_len': args.patch_len,
                'stride': args.stride
            }
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f" Summary saved to: {json_path}")