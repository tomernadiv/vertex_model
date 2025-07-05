
#!/bin/bash

# Set the simulation number
simulation_num=$1 # first argument passed to the script
echo "Running simulation number: $simulation_num"

# Parameter to iterate over (map simulation_num to param name)
declare -A sim_params
sim_params[1]="expansion_const"
sim_params[2]="push_out_force_strength"
sim_params[3]="shrinking_const"

# Possible values for each parameter (manually defined arrays since Bash has no float range)
expansion_const_values=($(seq 1.05 0.1 1.8))
push_out_force_strength_values=($(seq 1.0 0.2 3.4))
shrinking_const_values=($(seq 0.05 0.05 1.0))

# Get the parameter name
param_name=${sim_params[$simulation_num]}

# Define input and output paths
config_file="configs/simulation_${simulation_num}.py"
output_file="results/simulation_${simulation_num}/run_info.json"
mkdir -p "results"

# ========================
# 1. Activate base conda env
# ========================
# echo "------------------------------------------------------------------"
# module load miniconda/22.11.1_environmentally
# conda activate vertex_env
# pid=$!
# wait $pid

# ========================
# 2. Run simulations
# ========================
echo "Running simulations for parameter: $param_name"

# Get the corresponding values array
values_var="${param_name}_values[@]"
param_values=("${!values_var}")

# Prepare result file
result_file="results/simulation_${simulation_num}_final_results.txt"
echo -e "# Parameter: $param_name\n# Format: value\tfinal_window_area\tmax_velocity_at_10\tfinal_total_area" > "$result_file"


for value in "${param_values[@]}"; do
    echo "Running simulation with $param_name = $value"

    # Replace the line in the config file that sets the parameter (assumes `param_name = value` format)
    sed -i "s/^${param_name} *= *.*/${param_name} = ${value}/" "$config_file"
    pid=$!
    wait $pid
    
    # Run simulation
    python run_simulation.py False $simulation_num
    pid=$!
    wait $pid

    # Extract the final window area from the output
    final_window_area=$(grep "last_window_size" "$output_file" | awk '{print $NF}')
    final_window_area=${final_window_area%,}  # Remove trailing comma if present

    max_v=$(grep "max_at_10" "$output_file" | awk '{print $NF}')
    max_v=${max_v%,}  # Remove trailing comma if present

    final_total_area=$(grep "final_total_area" "$output_file" | awk '{print $NF}')
    final_total_area=${final_total_area%,}  # Remove trailing comma if present


    echo "Final window area: $final_window_area"
    echo "Max velocity at t=10: $max_v"
    echo "Final total area: $final_total_area"
    pid=$!
    wait $pid

    # Save result: value and window area
    echo -e "${value}\t${final_window_area}\t${max_v}\t${final_total_area}" >> "$result_file"
    pid=$!
    wait $pid
done

# plot the results
# python plot_changing_win_size.py "$simulation_num" "max_velocity_at_10"
python plot_changing_win_size.py "$simulation_num" "final_window_area"


# ========================
# 3. Save results
# ========================

echo "Results saved to $result_file"
echo "Done."