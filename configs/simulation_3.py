""""
    Neurons shrink 
    The nerons want to shrink and  other cells towards them.
"""
# mechanical properties
description = "Neurons Shrink"
forces = ['spring']
mu = 1                               # friction coefficient
shrinking_const = 0.59
expansion_const = 1
inner_border_layers = 1
push_out_force_strength = 0.0
neuron_spring_constant_marginal = 6        # spring between neighbouring vertices
neuron_spring_constant_boundary = 6         # spring between boundary vertices (vertices with 3 neighbours)
neuron_spring_constant_internal = 6      # spring between opposite vertices of the same cell
non_neuron_spring_constant_marginal = 1     # spring between neighbouring vertices
non_neuron_spring_constant_boundary = 1     # spring between boundary vertices (vertices with 3 neighbours)
non_neuron_spring_constant_internal = 1     # spring between opposite vertices of the same cell
line_tension_constant = 0.1                 # causes neurons to shrink
