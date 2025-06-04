""""
    Neurons shrink 
    The nerons want to shrink and  other cells towards them.
"""
# mechanical properties
forces = ['spring']
mu = 1                               # friction coefficient
shrinking_const = 0.5
expansion_const = 1
inner_border_layers = 0
push_out_force_strength = 0.0
spring_constant_marginal = 2         # spring between neighbouring vertices
spring_constant_boundary = 2         # spring between boundary vertices (vertices with 3 neighbours)
spring_constant_internal = 2         # spring between opposite vertices of the same cell
line_tension_constant = 0.1          # causes neurons to shrink
internal_rest_length = 2 * cell_initial_vertex_length * shrinking_const
boundary_rest_length = cell_initial_vertex_length * shrinking_const
marginal_rest_length = cell_initial_vertex_length * shrinking_const
non_neuron_internal_rest_length = 2 * cell_initial_vertex_length * expansion_const
non_neuron_boundary_rest_length = cell_initial_vertex_length * expansion_const
non_neuron_marginal_rest_length = cell_initial_vertex_length * expansion_const