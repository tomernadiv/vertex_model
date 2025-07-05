
#!/bin/bash

# Set the simulation number
simulation_num=$1 # first argument passed to the script
echo "Running simulation number: $simulation_num"


# Possible values for num_layers
num_layers_values=()
for ((i=28; i>=10; i--)); do
    num_layers_values+=($i)
done


# Compute window_size for each num_layers
window_size_values=()
for num_layers in "${num_layers_values[@]}"; do
    window_size=$((60 - 2 * num_layers))
    window_size_values+=($window_size)
done


# Get the parameter name
param_name = 'window_size'
config_param_name = 'num_layers'

# Define input and output paths
config_file="configs/morphology_config.py"
output_file="results/simulation_${simulation_num}/run_info.json"
mkdir -p "results"

echo "Running simulations for parameter: $param_name"

# Prepare result file
result_file="results/simulation_${simulation_num}_window_size_vs_final_area.txt"
echo -e "# Parameter: $param_name\n# Format: value\tfinal_window_area" > "$result_file"

# replace the number of windows to 1 in the config file
sed -i "s/^${num_frames}[[:space:]]*=[[:space:]]*.*/${num_frames} = 1/" "$config_file"

for i in "${!num_layers_values[@]}"; do
    num_layers=${num_layers_values[$i]}
    window_size=${window_size_values[$i]}

    # Replace the line in the config file that sets the parameter (assumes `param_name = value` format)
    sed -i "s/^${config_param_name} *= *.*/${config_param_name} = ${num_layers}/" "$config_file"
    pid=$!
    wait $pid
    
    # Run simulation
    python run_simulation.py False $simulation_num
    pid=$!
    wait $pid

    # Extract the final window area from the output
    final_window_area=$(grep "last_window_size" "$output_file" | awk '{print $NF}')
    final_window_area=${final_window_area%,}  # Remove trailing comma if present

    echo "Final window area: $final_window_area"
    pid=$!
    wait $pid

    # Save result: value and window area
    echo -e "${window_size}\t${final_window_area}" >> "$result_file"
    pid=$!
    wait $pid
done

# Reset the number of frames in the config file to 2
sed -i "s/^${num_frames}[[:space:]]*=[[:space:]]*.*/${num_frames} = 2/" "$config_file"


# plot the results
python plot_window_size_vs_final_area.py "$simulation_num"


# ========================
# 3. Save results
# ========================

echo "Results saved to $result_file"
echo "Done."