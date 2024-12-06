#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1

python validate.py --data nllb-seed --src eng_Latn --tgt xx --model_dir /mnt/storage/fking/models/seed_5687 --nllb_model 1.3B