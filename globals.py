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
cell_volume = 1
cell_initial_vertex_length = cell_radius
cell_initial_surface_area = 6 * 0.5 * ((cell_radius ** 2) * math.sqrt(3) / 2)
cell_initial_height = cell_volume / cell_initial_surface_area

# probs
neuron_prob = 0.5 # probability of a cell being a neuron

# mechanical properties
mu = 0.5                         # friction coefficient
spring_constant_marginal = 1        # spring between neighbouring vertices
spring_constant_boundary = 1        # spring between boundary vertices (vertices with 3 neighbours)
spring_constant_internal = 0      # spring between opposite vertices of the same cell
line_tension_constant = 0    # causes neurons to shrink
internal_rest_length = cell_radius * 2 * 0.7     # rest length of the internal spring
boundary_rest_length = cell_initial_vertex_length * 0.7     # rest length of the boundary spring
marginal_rest_length = (cell_initial_vertex_length * 0.7) 


# constrints
marginal_min_length = 0.1
internal_min_length = 0.1
boundary_min_length = 0.1

