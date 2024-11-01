#!/bin/bash
#SBATCH --job-name=basic_test
#SBATCH --output=basic_test.out
#SBATCH --error=basic_test.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=cpu

module load python/3.8
python hpctest.py