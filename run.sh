#!/bin/bash
#BSUB -J  1run1 Job name with sim number
#BSUB -q short                 # Queue name
#BSUB -R "rusage[mem=20000]"  # RAM
#BSUB -o out/%J.out           # Output file
#BSUB -e err/%J.err           # Error file

# Create output dirs
mkdir -p out err results

# Load conda and activate env
module load miniconda/22.11.1_environmentally
conda activate vertex_env

# Run your simulation script
# bash choose_parameters.sh 1 
# python run_simulation.py
# bash window_size_vs_area.sh 1 
# python plot_window_size_vs_final_area.py 1
# python plot_changing_win_size.py 1 "final_window_area"
python run_simulation.py 1