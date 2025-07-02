""""
    The inner border of the window of the non-neurons have cpnstant force to "push outwards" from the center of the window.  
    This border of cells should pull other non-nurons outwards.
"""

# mechanical properties
description = "Inner Border Push Out"
forces = ['spring' ,'push_out']
mu = 1                                       # friction coefficient
shrinking_const = 1
expansion_const = 1
inner_border_layers = 1
push_out_force_strength = 2.1
neuron_spring_constant_marginal = 3         # spring between neighbouring vertices
neuron_spring_constant_boundary = 3         # spring between boundary vertices (vertices with 3 neighbours)
neuron_spring_constant_internal = 3         # spring between opposite vertices of the same cell
non_neuron_spring_constant_marginal = 3    # spring between neighbouring vertices
non_neuron_spring_constant_boundary = 3         # spring between boundary vertices (vertices with 3 neighbours)
non_neuron_spring_constant_internal = 3         # spring between opposite vertices of the same cell
line_tension_constant = 0.1                 # causes neurons to shrink

# push out param
min_push_out_force = 0.1                   # minimum force to push outwards
push_out_decay_type = 'constant'              # lin, exp or constant
push_out_decay_constant_lin = 0.000001      
push_out_decay_constant_exp = 0.000001