#!/bin/bash
#BSUB -J sim_3   # Job name
#BSUB -q long                 # Queue name
#BSUB -R "rusage[mem=40000]"  # RAM
#BSUB -o out/%J.out           # Output file
#BSUB -e err/%J.err           # Error file

# Make sure the 'out' and 'err' directories exist
mkdir -p out err

# Activate your environment
module load miniconda/22.11.1_environmentally
conda activate vertex_env

# Run your Python script 
python run_simulation.py 3