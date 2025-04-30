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
        self.cells = []

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
        # boundary_edges, boundary_nodes  = self._find_boundary(self.graph)
        boundary_edges, boundary_nodes = [],[]
        #edges
        edge_attr = {edge: 'boundary' for edge in boundary_edges}
        nx.set_edge_attributes(self.graph, edge_attr, name='edge_type')
        # nodes
        boundary_attr = {node: True for node in boundary_nodes}
        nx.set_node_attributes(self.graph, False, 'boundary')  # Set default to False
        nx.set_node_attributes(self.graph, boundary_attr, 'boundary')  # Update boundary nodes to True

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
        # Reset all forces to zero  
        for node in self.graph.nodes:
            self.graph.nodes[node]['force'] = np.array([0.0, 0.0])

        #  iterate over each unique edge
        for v1, v2 in self.graph.edges:

            # iterate over all forces
            for force_name in forces:

                # compute force
                force = self._compute_force(force_name, v1, v2)  

                # sum up force only if vetex type is not boundary
                if not self.graph.nodes[v1]['boundary']:
                    self.graph.nodes[v1]['force'] += force
                
                if not self.graph.nodes[v2]['boundary']:
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

    def update_heights(self):
        """
        Update the volume of each cell based on the positions of its vertices.
        Aassume cell volume is constant, compute the new surface area, and update cell height
        """
        for cell in self.cells:
            hex_nodes = cell.get_nodes()
            pos = [self.graph.nodes[node]['pos'] for node in hex_nodes]

            # compute new surface area
            new_surface_area = 0
            for i in range(6):
                p1 = pos[i]
                p2 = pos[(i + 1) % 6]
                new_surface_area += 0.5 * abs(p1[0] * p2[1] - p2[0] * p1[1])
            
            # compute new height
            new_height = cell_volume / new_surface_area
            cell.update_height(new_height)                 

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

        for cell in self.cells:
            hex_nodes = cell.get_nodes()
            color_map = neurons_cmap if cell.is_neuron() else non_neurons_cmap
            color = cm.get_cmap(color_map)(cell.get_height() / 2)
            poly_coords = [pos[node] for node in hex_nodes]
            ax.fill(*zip(*poly_coords), color=color, alpha=0.5)

        if legend:
            neuron_patch = patches.Patch(color=cm.get_cmap(neurons_cmap)(0.5), label='Neuron')
            non_neuron_patch = patches.Patch(color=cm.get_cmap(non_neurons_cmap)(0.5), label='Non-neuron')
            ax.legend(handles=[neuron_patch, non_neuron_patch])


        ax.set_aspect('equal')
        ax.axis('off')

        if created_fig:
            plt.show()

    def _f_spring(self, v1, v2):
        """
        Calculate the spring force between two vertices.
        """
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        dist = math.sqrt(dx**2 + dy**2)

        # # Avoid division by zero
        # if dist == 0:
        #     return np.array([0.0, 0.0])
        
        # get spring constant accorfing to edge type
        edge_type = self._get_edge_type(v1, v2)
        spring_constant = globals()[f"spring_constant_{edge_type}"]
        min_length = globals()[f"{edge_type}_min_length"]
        rest_length = globals()[f"{edge_type}_rest_length"]
        if dist < min_length:
            # spring constrain - not compressing too much
            dist = min_length
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

            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            dist = math.sqrt(dx**2 + dy**2)

            # # Avoid division by zero
            # if dist == 0:
            #     return np.array([0.0, 0.0])

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
    
    def compute_total_energy(self):
        total_energy = 0.0

        for v1, v2 in self.graph.edges:
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length = math.sqrt(dx**2 + dy**2)
            
            edge_type = self._get_edge_type(v1, v2)
            spring_constant = globals()[f"spring_constant_{edge_type}"]
            rest_length = globals()[f"{edge_type}_rest_length"]

            # if (not (self.graph.nodes[v1]['neuron'] and self.graph.nodes[v2]['neuron'])):
            #     #not sure if this is the contribution of line tention
            #     total_energy += line_tension_constant * length
            
            # Spring energy: 0.5 * k * (l - l0)^2
            spring_energy = 0.5 * spring_constant * (length - rest_length)**2
            total_energy += spring_energy
        return total_energy




    # def _f_repulsion_internal_membrane(self, v1, v2):
    #     """
    #     Repulsive force between internal and marginal edges to prevent crossing.
    #     Applies when v1-v2 is an internal edge, and checks nearby marginal edges.
    #     """
    #     edge_type = self._get_edge_type(v1, v2)
    #     if edge_type != "internal":
    #         return np.array([0.0, 0.0])

    #     pos1  = np.array(self.graph.nodes[v1]['pos'])
    #     pos2  = np.array(self.graph.nodes[v2]['pos'])
    #     force = np.array([0.0, 0.0])

    #     for u, w, attrs in self.graph.edges(data=True):
    #         if (u, w) == (v1, v2) or (w, u) == (v1, v2):
    #             continue
    #         if attrs.get("edge_type") != "marginal":
    #             continue

    #         # get positions
    #         p1 = np.array(self.graph.nodes[u]['pos'])
    #         p2 = np.array(self.graph.nodes[w]['pos'])

    #         # shortest vector from internal edge to marginal edge
    #         closest_vec = self._shortest_vector_between_segments(pos1, pos2, p1, p2)
    #         dist = np.linalg.norm(closest_vec)

    #         if dist < repulsion_threshold and dist > 0:
    #             repulsion = repulsion_strength * (repulsion_threshold - dist) * (closest_vec / dist)
    #             force += repulsion

    #     return force

    # def _shortest_vector_between_segments(self, a1, a2, b1, b2):
    #     """
    #     Computes the shortest vector from segment a1-a2 to b1-b2.
    #     Returns the vector pointing from internal edge to marginal edge.
    #     """
    #     def project_point_on_segment(p, q1, q2):
    #         line_vec = q2 - q1
    #         t = np.dot(p - q1, line_vec) / np.dot(line_vec, line_vec)
    #         t = max(0, min(1, t))
    #         return q1 + t * line_vec

    #     # project a1 and a2 on b1-b2 and pick the closest
    #     proj1 = project_point_on_segment(a1, b1, b2)
    #     proj2 = project_point_on_segment(a2, b1, b2)
    #     d1 = a1 - proj1
    #     d2 = a2 - proj2

    #     return d1 if np.linalg.norm(d1) < np.linalg.norm(d2) else d2
