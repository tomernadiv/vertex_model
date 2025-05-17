import os
import sys
import numpy as np
import seaborn as sns
import torch
import networkx as nx
import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib import colors
import matplotlib.image as mpimg




# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# set plotting params
plt.rcParams['figure.figsize'] = (10, 6)
non_neurons_cmap = 'Oranges'
neurons_cmap = 'Blues'



### phyisical constants ###


# cell size
cell_radius = 1
cell_volume = 6 * 0.5 * ((cell_radius ** 2) * math.sqrt(3) / 2) # so that initial height will be 1
cell_initial_vertex_length = cell_radius
cell_initial_surface_area = 6 * 0.5 * ((cell_radius ** 2) * math.sqrt(3) / 2)
cell_initial_height = cell_volume / cell_initial_surface_area


# mechanical properties
mu = 1                               # friction coefficient
shrinking_const = 0.8
spring_constant_marginal = 2         # spring between neighbouring vertices
spring_constant_boundary = 2        # spring between boundary vertices (vertices with 3 neighbours)
spring_constant_internal = 2         # spring between opposite vertices of the same cell
line_tension_constant = 0.1          # causes neurons to shrink
internal_rest_length = 2 * cell_initial_vertex_length * shrinking_const
boundary_rest_length = cell_initial_vertex_length * shrinking_const
marginal_rest_length = cell_initial_vertex_length * shrinking_const
non_neuron_internal_rest_length = 2 * cell_initial_vertex_length
non_neuron_boundary_rest_length = cell_initial_vertex_length 
non_neuron_marginal_rest_length = cell_initial_vertex_length

# constraints
marginal_min_length = 0.1
internal_min_length = 0.1
boundary_min_length = 0.1

# for plotting
min_height = 10e-1
max_height = 10e1
min_area = 10e-1
max_area = 10e1

