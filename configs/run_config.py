import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# set plotting params
plt.rcParams['figure.figsize'] = (10, 6)