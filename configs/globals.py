from math import sqrt

non_neurons_cmap = 'Greens_r'
neurons_cmap = 'Reds_r'
inner_border_cmap = 'Blues_r'

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
