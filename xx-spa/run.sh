#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python newtok_train.py quechua-spanish nllb-quy-spa-newtok m
