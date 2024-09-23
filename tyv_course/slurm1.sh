#!/bin/sh
SBATCH -c 1                # Request 1 CPU core
SBATCH --gres=gpu:1        # Request one GPUs
SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 mins 
SBATCH -p dl               # Partition to submit to (should always be dl, for now)
SBATCH --mem=15G           # Request 15G of memory
SBATCH -o myoutput_%j.out  # File to which STDOUT will be written (%j inserts jobid)
SBATCH -e myerrors_%j.err  # File to which STDERR will be written (%j inserts jobid)

# Command you want to run on the cluster
# Notice, you must set-up testEval correctly as a conda virtual environment
# Calling this full path makes sure you are running the correct package versions
~/anaconda3/envs/testEnv/bin/python3.9 train_net.py