<img width="451" height="542" alt="Screenshot 2026-03-27 185707" src="https://github.com/user-attachments/assets/7488d9ee-1df6-4c34-8631-6a353f256dcb" />

 I've set up the PatchTST repo and got everything working with no package issues (finally). Tested it with 1 epoch on the supervised version and it runs smoothly.

## For you guys to also use it, run these commands

```bash
cd /scratch/zt1/project/msml612/user/jeffy22/MSML612_EnergyForecasting
source team_setup.sh
```

That's it. i made the srcipts to automatically:

- Activate the environment  
- Create `results/yourname/` and `logs/yourname/` folders for you  
- Set up all the paths  

*(You might see CUDA = False on the login node whihc is normal. The GPUs are shy and only show up when you ask nicely.)*

---

## Once you are done with it test it just in case

grab a GPU :

```bash
srun --partition=gpu --gres=gpu:a100:1 --time=00:30:00 --mem=16GB --cpus-per-task=4 --pty bash
```

run a quick 1-epoch test:

```bash
cd $CODE_DIR
python run_longExp.py \
  --is_training 1 \
  --model_id test \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --train_epochs 1 \
  --use_gpu True
```

If it prints numbers and doesn't scream at you, we're good. 

---

Don't forget to escape the GPU when you're done:

```bash
exit
```

That's it. You're ready to go. 

**P.S.** Don't break the shared environment. I spent too long on it. 😅

