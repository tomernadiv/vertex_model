from globals import *
from cell import Cell
from networkx.algorithms.boundary import edge_boundary



class Tissue:
    def __init__(self, cell_radius, num_rows, num_cols):
        self.cell_radius = cell_radius
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols
        self.graph = nx.Graph()
        self.cells: list[Cell] = []

        self._create_grid()

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

                is_neuron = row > (int(self.num_rows/2) - 1)
                is_neuron = True
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

    # def plot_tissue(self, ax=None, legend=False):

    #     # get positions
    #     pos = nx.get_node_attributes(self.graph, 'pos')

    #     created_fig = False
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(6, 6))
    #         created_fig = True
    #     ax.set_xlim(-1, self.num_cols * 3 / 2 + 1)
    #     ax.set_ylim(-1, self.num_rows * math.sqrt(3) + 1)
    #     # Draw edges grouped by type
    #     edge_colors = {
    #         'marginal': 'gray',
    #         'internal': 'red',
    #         'boundary': 'green'
    #     }
    #     for edge_type, color in edge_colors.items():
    #         edge_list = [
    #             (u, v) for u, v, d in self.graph.edges(data=True)
    #             if d.get('edge_type') == edge_type
    #         ]
    #         nx.draw_networkx_edges(
    #             self.graph, pos,
    #             edgelist=edge_list,
    #             edge_color=color,
    #             width=0.5,
    #             ax=ax
    #         )


    #     node_colors = [
    #         'green' if self.graph.nodes[node].get("boundary") else 'gray'
    #         for node in self.graph.nodes
    #     ]
    #     nx.draw_networkx_nodes(
    #         self.graph,
    #         pos,
    #         node_size=5,
    #         node_color=node_colors,
    #         ax=ax
    #     )

    #     for cell in self.cells:
    #         hex_nodes = cell.get_nodes()
    #         color_map = neurons_cmap if cell.is_neuron() else non_neurons_cmap
    #         color = cm.get_cmap(color_map)(cell.get_height() / 2)
    #         poly_coords = [pos[node] for node in hex_nodes]
    #         ax.fill(*zip(*poly_coords), color=color, alpha=0.5)

    #     if legend:
    #         neuron_patch = patches.Patch(color=cm.get_cmap(neurons_cmap)(0.5), label='Neuron')
    #         non_neuron_patch = patches.Patch(color=cm.get_cmap(non_neurons_cmap)(0.5), label='Non-neuron')
    #         ax.legend(handles=[neuron_patch, non_neuron_patch])


    #     ax.set_aspect('equal')
    #     ax.axis('off')

    #     if created_fig:
    #         plt.show()

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
        min_h, max_h = 0.1, 0.9

        for cell in self.cells:
            hex_nodes = cell.get_nodes()
            color_map = neurons_cmap if cell.is_neuron() else non_neurons_cmap
            norm_height = (cell.get_height() - min_h) / (max_h - min_h)  # Normalize height
            color = cm.get_cmap(color_map)(1 - norm_height)  # Reversed color mapping (higher = darker)
            poly_coords = [pos[node] for node in hex_nodes]
            ax.fill(*zip(*poly_coords), color=color, alpha=0.8)  # stronger color

        if legend:
            neuron_patch = patches.Patch(color=cm.get_cmap(neurons_cmap)(0.5), label='Neuron')
            non_neuron_patch = patches.Patch(color=cm.get_cmap(non_neurons_cmap)(0.5), label='Non-neuron')
            ax.legend(handles=[neuron_patch, non_neuron_patch])

        ax.set_aspect('equal')
        ax.axis('off')

        if created_fig:
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
        rest_length = globals()[f"{edge_type}_rest_length"]

        # Repulsion if distance is less than minimal allowed
        if dist < min_length:
            print(f"[Warning]: distance between {v1}, {v2} is below minimal ({dist:.3f} < {min_length})")
            # Apply a repulsive force to push apart strongly
            force_magnitude = spring_constant * (dist - rest_length)
            repulsion_strength = 1.5  # or higher if needed
            force_magnitude = repulsion_strength * spring_constant * (dist - min_length)
        else:
            force_magnitude = spring_constant * (dist - rest_length)


        force_vector = np.array([force_magnitude * dx / dist, force_magnitude * dy / dist])
        return force_vector
    
    def _f_line_tension(self, v1, v2):
        """
        Calculate the line tension force between two vertices,
        only if both vertices are neurons, and the edge_type is not "internal".
        """
        edge_type = self._get_edge_type(v1, v2)

        # don't compute on internal or boundary edges?? 
        if ((self.graph.nodes[v1]['neuron'] and self.graph.nodes[v2]['neuron'])) and (edge_type == "marginal"):

            dx, dy, dist = self._get_distances(v1,v2)

            # Avoid division by zero
            if dist == 0:
                raise RuntimeError(f"distance of nodes: {v1}, {v2} is zero!")

            unit_vector = np.array([dx / dist, dy / dist])
            force_vector = line_tension_constant * unit_vector

            return force_vector
        else:
            return np.array([0.0, 0.0])

    
    def _get_edge_type(self, v1, v2):
        edge_data = self.graph.get_edge_data(v1, v2)
        if edge_data is not None:
            edge_type = edge_data.get('edge_type')
        return edge_type
    
    def update_heights(self):
        """
        Update the volume of each cell based on the positions of its vertices.
        Aassume cell volume is constant, compute the new surface area, and update cell height
        """
        for cell in self.cells:
            hex_nodes = cell.get_nodes()
            #pos = [self.graph.nodes[node]['pos'] for node in hex_nodes]

            # compute new surface area
            new_surface_area = 0
            for i in range(6):
<<<<<<< Updated upstream
                v1 = hex_nodes[i]
                v2 = hex_nodes[(i + 1) % 6]
                p1 = np.array(self.graph.nodes[v1]['pos'])
                p2 = np.array(self.graph.nodes[v2]['pos'])
=======
                p1 = np.array(pos[i])
                p2 = np.array(pos[(i + 1) % 6])
>>>>>>> Stashed changes
                new_surface_area += 0.5 * abs(p1[0] * p2[1] - p2[0] * p1[1])
            
            # compute new height
            new_height = cell_volume / new_surface_area

            cell.update_height(new_height)    

    def _get_distances(self, v1, v2):
        p1 = np.array(self.graph.nodes[v1]['pos'])
        p2 = np.array(self.graph.nodes[v2]['pos'])
        dx, dy = p2 - p1
        dist = math.sqrt(dx**2 + dy**2)
        return dx, dy, dist
    
    def compute_total_energy(self):
        total_energy = 0.0

        for v1, v2 in self.graph.edges:
            dx,dy,dist = self._get_distances(v1,v2)
            
            edge_type = self._get_edge_type(v1, v2)
            spring_constant = globals()[f"spring_constant_{edge_type}"]
            rest_length = globals()[f"{edge_type}_rest_length"]

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