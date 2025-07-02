
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
# expansion_const_values=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9)
expansion_const_values=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
push_out_force_strength_values=(2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0)
shrinking_const_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Get the parameter name
param_name=${sim_params[$simulation_num]}

# Define input and output paths
config_file="configs/simulation_${simulation_num}.py"
output_file="results/simulation_${simulation_num}/run_info.json"
mkdir -p "results"

# ========================
# 1. Activate base conda env
# ========================
echo "------------------------------------------------------------------"
echo "Activating conda environment: vertex_model"
conda activate vertex_model
pid=$!
wait $pid

# ========================
# 2. Run simulations
# ========================
echo "Running simulations for parameter: $param_name"

# Get the corresponding values array
values_var="${param_name}_values[@]"
param_values=("${!values_var}")

# Prepare result file
result_file="results/simulation_${simulation_num}_final_results.txt"
echo -e "# Parameter: $param_name\n# Format: value\tfinal_window_area\tmax_velocity_at_10" > "$result_file"


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


    echo "Final window area: $final_window_area"
    echo "Max velocity at t=10: $max_v"
    pid=$!
    wait $pid

    # Save result: value and window area
    echo -e "${value}\t${final_window_area}\t${max_v}" >> "$result_file"
    pid=$!
    wait $pid
done

# ========================
# 3. Save results
# ========================

echo "Results saved to $result_file"
echo "Done."


# simulation_num = 1
# # parameter to iterate over
# sim_params = {1: "expansion_const", 2: "push_out_force_strength", 3: "shrinking_const"}

# # possible values for each parameter
# param_values = {   
#     "expansion_const": range(1.0, 3.0, 0.1),
#     "push_out_force_strength": range(1.0, 7.0, 0.25),
#     "shrinking_const": range(0.1, 1.0, 0.1)
# }

# # get config file
# config_file = "configs/simulation_${simulation_num}.py"
# output_file = results/simulation_${simulation_num}/run_info.json


# # ========================
# # 1. Activate base conda env
# # ========================
#     echo "------------------------------------------------------------------"
#     echo "Activating conda environment: nova_nova"
#     module load
#     ml miniconda
#     conda activate vertex_model

# # ========================
# # 2. run simulations
# # ========================

# final_window_areas = []
# # iterate over values
# for value in param_values[sim_params[simulation_num]]:
#     # update config file with the new value
#     grep "${sim_params[simulation_num]}" $config_file | sed -i "s/=.*/= ${value}/" $config_file

#     pid=$!
#     wait $pid

#     # run the simulation with the updated config
#     python run_simulation.py 

#     pid=$!
#     wait $pid

#     # extract the final window area from the output
#     final_window_area=$(grep "last_window_size" $output_file | awk '{print $NF}')

#     pid=$!
#     wait $pid

# # ========================
# # 3. extract results to one plot
# # ========================

# echo final_window_areas >> results/simulation_${simulation_num}_final_window_areas.txt

# echo "done."