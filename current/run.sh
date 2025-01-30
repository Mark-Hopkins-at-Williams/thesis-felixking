#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-12:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1

python make_dataframe.py 6193 /mnt/storage/fking/data/seed_line_by_line /mnt/storage/fking/data/seed/seed
