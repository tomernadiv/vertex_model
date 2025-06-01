from math import sqrt

non_neurons_cmap = 'Greens'
neurons_cmap = 'Reds'

### phyisical constants ###
# cell size
cell_radius = 1
cell_volume = 6 * 0.5 * ((cell_radius ** 2) * sqrt(3) / 2) # so that initial height will be 1
cell_initial_vertex_length = cell_radius
cell_initial_surface_area = 6 * 0.5 * ((cell_radius ** 2) * sqrt(3) / 2)
cell_initial_height = cell_volume / cell_initial_surface_area

# constraints
marginal_min_length = 0.1
internal_min_length = 0.1
boundary_min_length = 0.1

# for plotting
min_height = 10e-1 #1
max_height = 20e0 #10
min_area = 10e-1 #1
max_area = 12e0 #12

