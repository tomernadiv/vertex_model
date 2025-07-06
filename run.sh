#!/bin/bash
#BSUB -J 3              # Job name with sim number
#BSUB -q long                 # Queue name
#BSUB -R "rusage[mem=10000]"  # RAM
#BSUB -o out/%J.out           # Output file
#BSUB -e err/%J.err           # Error file

# Create output dirs
mkdir -p out err results

# Load conda and activate env
module load miniconda/22.11.1_environmentally
conda activate vertex_env

# Run your simulation script
# bash choose_parameters.sh 2



# python run_simulation.py
# python plot_window_size_vs_final_area.py 1
# python plot_changing_win_size.py 1 "final_window_area"
# python run_simulation.py 3
# bash choose_parameters.sh 1
bash window_size_vs_area.sh 3
