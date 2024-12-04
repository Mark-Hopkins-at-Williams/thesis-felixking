#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python train.py --src eng --tgt xx --csv /mnt/storage/fking/data/seed/all.csv --model_dir /mnt/storage/fking/models/nllb-seed-eng-xx-v2 --nllb_model 1.3B