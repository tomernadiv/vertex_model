from globals import *
from cell import Cell
from networkx.algorithms.boundary import edge_boundary
import neuron_initiation 


class Tissue:
    def __init__(self, cell_radius, num_rows, num_cols, n_init_func = "half_tissue", num_out_layers=0):
        self.cell_radius = cell_radius
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols
        self.graph = nx.Graph()
        self.cells: list[Cell] = []
        self.n_init_func = n_init_func
        self.num_out_layers = num_out_layers

        self._create_grid()

    
    def _create_grid(self):
        dx = 3/2 * self.cell_radius
        dy = math.sqrt(3) * self.cell_radius
        node_cache = {}

        def round_pos(x, y):
            return (round(x, 5), round(y, 5))

        # generate grid 
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # hexagon centroid coordinates
                cx = col * dx
                cy = row * dy
                if col % 2 == 1:
                    cy += dy / 2

                # generate hexagon nodes
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

                # add marginal edges between the hexagonal nodes, no self-loops (represents membrane)
                for i in range(6):
                    n1 = hex_nodes[i]
                    n2 = hex_nodes[(i + 1) % 6]
                    if n1 != n2:
                        self.graph.add_edge(n1, n2, edge_type='marginal')  

                # add internal edges between 2 extreme nodes in the hexagon (represents "volume")                        
                internal_pairs = [(0, 3), (1,4), (2, 5)]
                for i, j in internal_pairs:
                    n1 = hex_nodes[i]
                    n2 = hex_nodes[j]
                    if n1 != n2 and not self.graph.has_edge(n1, n2):
                        self.graph.add_edge(n1, n2, edge_type='internal')

                cell_index = int(row * self.num_cols + col)
                height = cell_initial_height                # can be modified later with a smarter logic
                
                is_neuron = (neuron_initiation.__dict__[self.n_init_func](row, self.num_rows)) and (not neuron_initiation.outline(self.num_out_layers, row, self.num_rows, col, self.num_cols))
                

                if is_neuron:
                    for node in hex_nodes:
                        self.graph.nodes[node]['neuron'] = True # set neuron flag for all vertices of the cell
                else:
                    for node in hex_nodes:
                        self.graph.nodes[node]['neuron'] = False


                # Create a Cell object and add it to the stack
                self.cells.append(Cell(cell_index, hex_nodes, height, is_neuron))
        
        # update metadata for boundry nodes and edges 
        boundary_edges, boundary_nodes  = self._find_boundary(self.graph)
        #boundary_edges, boundary_nodes = [],[]
        #edges
        edge_attr = {edge: 'boundary' for edge in boundary_edges}
        nx.set_edge_attributes(self.graph, edge_attr, name='edge_type')
        # nodes
        boundary_attr = {node: True for node in boundary_nodes}
        nx.set_node_attributes(self.graph, False, 'boundary')  # Set default to False
        nx.set_node_attributes(self.graph, boundary_attr, 'boundary')  # Update boundary nodes to True
    
    def _find_boundary(self, G):
        """Find the boundary edges of a hexagonal graph based on 'marginal' edges."""
        boundary_edges = []
        boundary_nodes = set()
        for u, v, attrs in G.edges(data=True):
            edge_type = attrs.get('edge_type')
            if edge_type != 'marginal':
                continue  # Skip non-marginal edges

            # Count 'marginal' edges connected to u
            marginal_degree_u = sum(
                1 for nbr in G.neighbors(u)
                if G[u][nbr].get('edge_type') == 'marginal'
            )

            # Count 'marginal' edges connected to v
            marginal_degree_v = sum(
                1 for nbr in G.neighbors(v)
                if G[v][nbr].get('edge_type') == 'marginal'
            )

            # If either node has fewer than 3 'marginal' connections, or both has exaactly 5 connections it's a boundary edge
            if (marginal_degree_u < 3 or marginal_degree_v < 3) or (G.degree(u) == 5 and G.degree(v) == 5):
                boundary_edges.append((u, v))
                boundary_nodes.update([u, v])

        return boundary_edges, boundary_nodes

    def plot_tissue(self, ax=None, legend=False):
        # get positions
        pos = nx.get_node_attributes(self.graph, 'pos')

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            created_fig = True
        ax.set_xlim(-1, self.num_cols * 3 / 2 + 1)
        ax.set_ylim(-1, self.num_rows * math.sqrt(3) + 1)

        # Draw edges grouped by type
        edge_colors = {
            'marginal': 'gray',
            'internal': 'red',
            'boundary': 'green'
        }
        for edge_type, color in edge_colors.items():
            edge_list = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if d.get('edge_type') == edge_type
            ]
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=edge_list,
                edge_color=color,
                width=0.5,
                ax=ax
            )

        node_colors = [
            'green' if self.graph.nodes[node].get("boundary") else 'gray'
            for node in self.graph.nodes
        ]
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=5,
            node_color=node_colors,
            ax=ax
        )
    

        # Compute global min and max height for normalization

        log_min = np.log10(min_height)
        log_max = np.log10(max_height)
        norm = colors.Normalize(vmin=log_min, vmax=log_max)

        for cell in self.cells:
            node_keys = cell.get_nodes()
            node_positions = [self.graph.nodes[key]['pos'] for key in node_keys]
            color_map = neurons_cmap if cell.is_neuron() else non_neurons_cmap

            height = cell.get_height()
            safe_height = max(height, min_height)  # prevent log10 errors
            log_height = np.log10(safe_height)

            color = cm.get_cmap(color_map)(norm(log_height))
            ax.fill(*zip(*node_positions), color=color, alpha=0.8)


        if legend:
            neuron_patch = patches.Patch(color=cm.get_cmap(neurons_cmap)(0.5), label='Neuron')
            non_neuron_patch = patches.Patch(color=cm.get_cmap(non_neurons_cmap)(0.5), label='Non-neuron')
            ax.legend(handles=[neuron_patch, non_neuron_patch])

        ax.set_aspect('equal')
        ax.axis('off')

        if created_fig:
            plt.show()

    def plot_heights_distribution(self, ax=None, bins=30, log_scale=True):
        """
        Plot the distribution of cell heights using seaborn.
        """
        heights = [cell.get_height() for cell in self.cells]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(heights, bins=bins, kde=False, stat='probability', color='blue', alpha=0.6, ax=ax)
        ax.set_xlabel('Cell Height')
        ax.set_ylabel('proportion')
        ax.set_title('Distribution of Cell Heights')
        ax.set_ylim(0, 1)
        ax.set_xlim(min_height, max_height)

        if log_scale:
            ax.set_xscale('log')
            min_height_log, max_height_log = np.log10(min_height), np.log10(max_height)
            ax.set_xlim(min_height_log, max_height_log)
            ax.set_title('Distribution of Cell Heights (Log Scale)')

        plt.show()


    
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

        #  iterate over each unique edge
        for v1, v2 in self.graph.edges:

            # iterate over all forces
            for force_name in forces:

                # compute force
                force = self._compute_force(force_name, v1, v2)  

                # add forces
                self.graph.nodes[v1]['force'] += force
                self.graph.nodes[v2]['force'] -= force

    def update_positions(self, dt=1):
        """
        Update the position of each vertex based on the computed forces multiplied by a constant.
        - compute velocity, v = mu * f
        - update position, x = x + v * dt
        """
        for node in self.graph.nodes:

            # get position and force
            pos = np.array(self.graph.nodes[node]['pos'], dtype=float)
            force = np.array(self.graph.nodes[node]['force'], dtype=float)

            # compute velocity
            velocity = mu * force

            # update position
            new_pos = pos + velocity * dt

            self.graph.nodes[node]['pos'] = tuple(new_pos)

    def _f_spring(self, v1, v2):
        """
        Calculate the spring force between two vertices.
        """
        dx, dy, dist = self._get_distances(v1,v2)
        # Avoid division by zero
        if dist == 0:
            raise RuntimeError(f"distance of nodes: {v1}, {v2} is zero!")
        
        # get spring constant accorfing to edge type
        edge_type = self._get_edge_type(v1, v2)
        spring_constant = globals()[f"spring_constant_{edge_type}"]
        min_length = globals()[f"{edge_type}_min_length"]

        rest_length = self._get_rest_length(v1, v2, edge_type)
        force_magnitude = spring_constant * (dist - rest_length) + 1/(dist - min_length) # adding repulsion force 


        force_vector = np.array([force_magnitude * dx / dist, force_magnitude * dy / dist])
        return force_vector

    def _get_rest_length(self, v1, v2, edge_type):
        if self._is_nueron_edge(v1, v2):
            rest_length = globals()[f"{edge_type}_rest_length"]
        else:
            rest_length = globals()[f"non_neuron_{edge_type}_rest_length"]
        return rest_length
    
    def _f_line_tension(self, v1, v2):
        """
        Calculate the line tension force between two vertices,
        only if both vertices are neurons, and the edge_type is not "internal".
        """
        edge_type = self._get_edge_type(v1, v2)

        # don't compute on internal or boundary edges?? 
        if ((self._is_nueron_edge(v1, v2))) and (edge_type == "marginal"):

            dx, dy, dist = self._get_distances(v1,v2)

            # Avoid division by zero
            if dist == 0:
                raise RuntimeError(f"distance of nodes: {v1}, {v2} is zero!")

            unit_vector = np.array([dx / dist, dy / dist])
            force_vector = line_tension_constant * unit_vector

            return force_vector
        else:
            return np.array([0.0, 0.0])
        
    def _is_nueron_edge(self, v1, v2):
        return (self.graph.nodes[v1]['neuron'] and self.graph.nodes[v2]['neuron'])
 
    def _get_edge_type(self, v1, v2):
        edge_data = self.graph.get_edge_data(v1, v2)
        if edge_data is not None:
            edge_type = edge_data.get('edge_type')
        return edge_type
    
    def update_heights(self):
        """
        Update the volume of each cell based on the positions of its vertices.
        Assuming conservation of volume.
        """
        for cell in self.cells:

            node_keys = cell.get_nodes()
            node_positions = [self.graph.nodes[key]['pos'] for key in node_keys]

            # formula for polygon area
            x, y = zip(*node_positions)
            area = 0.5 * abs(sum(x[i]*y[(i+1)%6] - x[(i+1)%6]*y[i] for i in range(6)))

            new_height = cell_volume / area  
            cell.update_height(new_height)

    def compute_total_area(self):
        """
        Compute the total area of the tissue.
        """
        total_area = 0
        for cell in self.cells:

            node_keys = cell.get_nodes()
            node_positions = [self.graph.nodes[key]['pos'] for key in node_keys]

            # formula for polygon area
            x, y = zip(*node_positions)
            area = 0.5 * abs(sum(x[i]*y[(i+1)%6] - x[(i+1)%6]*y[i] for i in range(6)))

            total_area += area 
        return total_area

    def _get_distances(self, v1, v2):
        def smart_round(val):
            """
            if the number is close to an integer - round to the integer
            """
            return round(val) if abs(val - round(val)) < 1e-3  else val #round(val, 3)
    
        p1 = np.array(self.graph.nodes[v1]['pos'])
        p2 = np.array(self.graph.nodes[v2]['pos'])
        dx, dy = p2 - p1
        dist = math.sqrt(dx**2 + dy**2)

        dx = smart_round(dx)
        dy = smart_round(dy)
        dist = smart_round(dist)
        return dx, dy, dist
    
    def compute_total_energy(self):
        total_energy = 0.0

        for v1, v2 in self.graph.edges:
            dx,dy,dist = self._get_distances(v1,v2)
            
            edge_type = self._get_edge_type(v1, v2)
            spring_constant = globals()[f"spring_constant_{edge_type}"]
            rest_length = self._get_rest_length(v1, v2, edge_type)

            # if (not (self.graph.nodes[v1]['neuron'] and self.graph.nodes[v2]['neuron'])):
            #     #not sure if this is the contribution of line tention
            #     total_energy += line_tension_constant * length
            
            # Spring energy: 0.5 * k * (l - l0)^2
            spring_energy = 0.5 * spring_constant * (dist - rest_length)**2
            total_energy += spring_energy
        return total_energy


    #### for debugging cell height colors:
    def compare_opposite_corner_heights(self):
        top_right_index = self.num_cols - 1
        bottom_left_index = (self.num_rows - 1) * self.num_cols

        top_right_cell = self.cells[top_right_index]
        bottom_left_cell = self.cells[bottom_left_index]

        h_top_right = top_right_cell.get_height()
        h_bottom_left = bottom_left_cell.get_height()

        print(f"Top-right cell (index {top_right_index}) height: {h_top_right:.3f}")
        print(f"Bottom-left cell (index {bottom_left_index}) height: {h_bottom_left:.3f}")
        print(f"Height difference (TR - BL): {h_top_right - h_bottom_left:.3f}")

        return h_top_right, h_bottom_left