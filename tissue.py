from globals import *
from cell import Cell

class Tissue:
    def __init__(self, cell_radius, num_rows, num_cols):
        self.cell_radius = cell_radius
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols
        self.graph = nx.Graph()
        self.cells = []

        self._create_grid()

    def _create_grid(self):
        dx = 3/2 * self.cell_radius
        dy = math.sqrt(3) * self.cell_radius
        node_cache = {}

        def round_pos(x, y):
            return (round(x, 5), round(y, 5))

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cx = col * dx
                cy = row * dy
                if col % 2 == 1:
                    cy += dy / 2

                hex_nodes = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    x = cx + self.cell_radius * math.cos(angle)
                    y = cy + self.cell_radius * math.sin(angle)
                    pos = round_pos(x, y)
                    if pos not in node_cache:
                        self.graph.add_node(pos, pos=pos, neuron=False, force=np.array([0.0, 0.0]))
                        node_cache[pos] = pos
                    hex_nodes.append(node_cache[pos])

                # add edges between the hexagonal nodes, no self-loops
                for i in range(6):
                    n1 = hex_nodes[i]
                    n2 = hex_nodes[(i + 1) % 6]
                    if n1 != n2:
                        self.graph.add_edge(n1, n2)                             

                cell_index = int(row * self.num_cols + col)
                height = cell_initial_height                # can be modified later with a smarter logic
                is_neuron = random.random() < neuron_prob   # probability of being a neuron

                if is_neuron:
                    for node in hex_nodes:
                        self.graph.nodes[node]['neuron'] = True # set neuron flag for all vertices of the cell
                else:
                    for node in hex_nodes:
                        self.graph.nodes[node]['neuron'] = False


                # Create a Cell object and add it to the stack
                self.cells.append(Cell(cell_index, hex_nodes, height, is_neuron))

    def _compute_force(self, force_name: str, v1, v2):
        """
        Compute the force between two vertices using the specified force name.
        The force name should match the method name in the class.
        """
        force_method = getattr(self, f"_f_{force_name}")
        return force_method(v1, v2)
    
    def compute_all_forces(self, forces: list):
        """
        Compute the sum of forces acting on the vertices of the graph.
        """
       # iterate on nodes, then iterate on their neighbors and compute forces
        for node in self.graph.nodes:
            for neighbor in self.graph.neighbors(node):
                for force_name in forces:
                    force = self._compute_force(force_name, node, neighbor)
                    self.graph.nodes[node]['force'] += force


    def plot_tissue(self, legend=False, timestamp=None):

        # get positions
        pos = nx.get_node_attributes(self.graph, 'pos')

        # draw base graph
        fig, ax = plt.subplots(figsize=(6,6))
        nx.draw_networkx_edges(self.graph, pos,
                            edge_color='gray',
                            width=0.5,
                            ax=ax)
        nx.draw_networkx_nodes(self.graph, pos,
                            node_size=5,
                            node_color='gray',
                            ax=ax)

        # iterate on all cells, color them according to their height
        for cell in self.cells:
            if not cell.is_neuron():
                hex_nodes = cell.get_nodes()
                color = cm.get_cmap(non_neurons_cmap)(cell.get_height()/2)
                plt.fill(*zip(*[pos[node] for node in hex_nodes]), color=color, alpha=0.5)

            else:
                hex_nodes = cell.get_nodes()
                color = cm.get_cmap(neurons_cmap)(cell.get_height()/2)
                plt.fill(*zip(*[pos[node] for node in hex_nodes]), color=color, alpha=0.5)

        # add legend
        if legend:
            neuron_patch = patches.Patch(color=cm.get_cmap(neurons_cmap)(0.5), label='Neuron')
            non_neuron_patch = patches.Patch(color=cm.get_cmap(non_neurons_cmap)(0.5), label='Non-neuron')
            plt.legend(handles=[neuron_patch, non_neuron_patch])

        # add title
        if timestamp is not None:
            plt.title(f"Timestamp: {timestamp}")

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _f_spring(self, v1, v2):
        """
        Calculate the spring force between two vertices.
        """
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        dist = math.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        if dist == 0:
            return np.array([0.0, 0.0])
        
        force_magnitude = spring_constant * (dist - cell_initial_vertex_length)
        force_vector = np.array([force_magnitude * dx / dist, force_magnitude * dy / dist])
        return force_vector
    
    def _f_line_tension(self, v1, v2):
        """
        Calculate the line tension force between two vertices,
        only if both vertices are neurons.
        """
        if not (self.graph.nodes[v1]['neuron'] and self.graph.nodes[v2]['neuron']):
            return np.array([0.0, 0.0])

        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        dist = math.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        if dist == 0:
            return np.array([0.0, 0.0])

        unit_vector = np.array([dx / dist, dy / dist])
        force_vector = line_tension_constant * unit_vector

        return force_vector
    

    def update_position(self):
        """
        Update the position of each vertex based on the computed forces.
        Dont forget to reset the forces after each update.
        """
        pass