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
                        self.graph.add_node(pos, pos=pos, neuron=False)
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

                # Create a Cell object and add it to the stack
                self.cells.append(Cell(cell_index, hex_nodes, height, is_neuron))


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

