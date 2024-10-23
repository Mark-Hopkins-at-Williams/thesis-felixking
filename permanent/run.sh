#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1

python finetune.py --data nllb-seed --dev_src lij_Latn --dev_tgt eng_Latn --model_dir /mnt/storage/fking/models/seed_$SLURM_JOB_ID
