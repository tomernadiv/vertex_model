non_neurons_cmap = 'hot_r'
neurons_cmap = 'hot_r'
inner_border_cmap = 'winter'
base_cmap = 'coolwarm'



### phyisical constants ###

# Parameters Taken From Real Experiments:
X_AXIS_LENGTH = 175   # [micro meter]

# constraints
marginal_min_length = 0.1
internal_min_length = 0.1
boundary_min_length = 0.1
cell_initial_vertex_length = 1
internal_rest_length = 2 * cell_initial_vertex_length * shrinking_const
boundary_rest_length = cell_initial_vertex_length 
marginal_rest_length = cell_initial_vertex_length * shrinking_const
non_neuron_internal_rest_length = 2 * cell_initial_vertex_length * expansion_const
non_neuron_boundary_rest_length = cell_initial_vertex_length 
non_neuron_marginal_rest_length = cell_initial_vertex_length * expansion_const
