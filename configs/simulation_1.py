""""
    all non-nuerons want to expand simultaneously.
    The outer cells should move the fastest.  
"""

# mechanical properties
description = "Non-Nuerons Expand"
forces = ['spring']
mu = 1                               # friction coefficient
shrinking_const = 1
expansion_const = 2.0
inner_border_layers = 1
push_out_force_strength = 0.0
neuron_spring_constant_marginal = 1         # spring between neighbouring vertices
neuron_spring_constant_boundary = 5         # spring between boundary vertices (vertices with 3 neighbours)
neuron_spring_constant_internal = 1         # spring between opposite vertices of the same cell
non_neuron_spring_constant_marginal = 1     # spring between neighbouring vertices
non_neuron_spring_constant_boundary = 5         # spring between boundary vertices (vertices with 3 neighbours)
non_neuron_spring_constant_internal = 1         # spring between opposite vertices of the same cell
line_tension_constant = 0.1          # causes neurons to shrink
