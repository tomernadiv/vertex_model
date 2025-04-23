import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns
import torch
import networkx as nx
import random
import math



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
cell_initial_vertex_length = math.sqrt(3) * cell_radius
cell_initial_surface_area = 3 * math.sqrt(3) / 2 * cell_initial_vertex_length**2
cell_initial_height = cell_volume / cell_initial_surface_area

# probs
neuron_prob = 0.5 # probability of a cell being a neuron

# mechanical properties
mu = 0.2                            # friction coefficient
spring_constant_marginal = 2      # spring between neighbouring vertices
spring_constant_boundary = 0      # spring between boundary vertices (vertices with 3 neighbours)
spring_constant_internal = 1      # spring between opposite vertices of the same cell
line_tension_constant = 1       # causes neurons to shrink
