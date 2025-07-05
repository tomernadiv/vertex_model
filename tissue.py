from cell import Cell
import neuron_initiation 
from configs.imports import *
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

class Tissue:
    def __init__(self, globals_config_path, simulation_config_path, morphology_config_path):
        self._init_simulation_properties(globals_config_path, simulation_config_path, morphology_config_path)
        self.num_cells = self.num_rows * self.num_cols
        self.graph = nx.Graph()
        self.cells: list[Cell] = []
        self._create_grid()
        self.initial_height = self.cells[0].get_height()
        self.initial_area = self.cells[0].get_area()
        self.time_step = 1 

    def zero_time_step(self):
        """
        Reset the time step to zero.
        """
        self.time_step = 1
    
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
                        self.graph.add_node(pos, pos=pos, neuron=False, force=np.array([0.0, 0.0]), row=row, col=col, index=i)
                        node_cache[pos] = pos
                    hex_nodes.append(node_cache[pos])

                # add marginal edges between the hexagonal nodes, no self-loops (represents membrane)
                for i in range(6):
                    n1 = hex_nodes[i]
                    n2 = hex_nodes[(i + 1) % 6]
                    if n1 != n2:
                        self.graph.add_edge(n1, n2, edge_type='marginal')  

                # add internal edges between 2 extreme nodes in the hexagon (represents "volume")                        
                internal_pairs = [ (0, 2), (2,4), (4,0), (1,3), (3,5), (5,1)] # david star, for diagonal: (0, 3), (1,4), (2, 5)
                for i, j in internal_pairs:
                    n1 = hex_nodes[i]
                    n2 = hex_nodes[j]
                    if n1 != n2 and not self.graph.has_edge(n1, n2):
                        self.graph.add_edge(n1, n2, edge_type='internal')

                cell_index = int(row * self.num_cols + col)
                height = self.cell_initial_height                # can be modified later with a smarter logic
                
                # check if nueron or not 
                is_neuron = self._init_cell(row, col)
                self._set_cell_attr(hex_nodes, 'neuron', is_neuron)

                # add inner layer if needed
                if self.inner_border_layers > 0:
                    is_border = neuron_initiation.inner_outline(self.num_layers,self.num_frames, row,self.num_rows,col ,self.num_cols,self.inner_border_layers)
                else:
                    is_border = (False, None)

                area = self.config_dict['cell_initial_surface_area']
                # Create a Cell object and add it to the stack
                self.cells.append(Cell(cell_index, hex_nodes, height, area, is_neuron, is_border))
        
        # update metadata for boundry nodes and edges 
        boundary_edges, boundary_nodes  = self._find_boundary(self.graph)

        #edges
        edge_attr = {edge: 'boundary' for edge in boundary_edges}
        nx.set_edge_attributes(self.graph, edge_attr, name='edge_type')
        # nodes
        boundary_attr = {node: True for node in boundary_nodes}
        nx.set_node_attributes(self.graph, False, 'boundary')  # Set default to False
        nx.set_node_attributes(self.graph, boundary_attr, 'boundary')  # Update boundary nodes to True

    def _set_cell_attr(self, hex_nodes, attr, value):
        for node in hex_nodes:
            self.graph.nodes[node][attr] = value

    def _init_simulation_properties(self, globals_config_path, simulation_config_path, morph_globals_path):
        # Load global and morphology constants
        simulation_consts = runpy.run_path(simulation_config_path)
        morph_consts = runpy.run_path(morph_globals_path)
        combined_namespace = {**simulation_consts, **morph_consts}

        # add globals config 
        with open(globals_config_path, 'r') as f:
            globals_data = f.read()
        exec(globals_data, combined_namespace)

        # initiate cell sizes parameters
        cell_radius = combined_namespace['X_AXIS_LENGTH'] / (1.5 * combined_namespace['num_cols'])  # radius of a cell in the hexagonal grid
        cell_volume = 6 * 0.5 * ((cell_radius ** 2) * math.sqrt(3) / 2)                   # so that initial height will be 1
        cell_initial_vertex_length = cell_radius
        cell_initial_surface_area = 6 * 0.5 * ((cell_radius ** 2) * math.sqrt(3) / 2)
        cell_initial_height = cell_volume / cell_initial_surface_area
        shrinking_const = combined_namespace['shrinking_const']
        expansion_const = combined_namespace['expansion_const']

        # add all to combined namespace
        combined_namespace.update({
            'cell_radius': cell_radius,
            'cell_volume': cell_volume,
            'cell_initial_vertex_length': cell_initial_vertex_length,
            'cell_initial_surface_area': cell_initial_surface_area,
            'cell_initial_height': cell_initial_height,
            'internal_rest_length': math.sqrt(3) * cell_initial_vertex_length * shrinking_const,
            'boundary_rest_length': cell_initial_vertex_length,
            'marginal_rest_length': cell_initial_vertex_length * shrinking_const,
            'non_neuron_internal_rest_length': math.sqrt(3) * cell_initial_vertex_length * expansion_const,
            'non_neuron_boundary_rest_length': cell_initial_vertex_length,
            'non_neuron_marginal_rest_length':cell_initial_vertex_length * expansion_const
        })

        # Assign all variables as self attributes
        for key, value in combined_namespace.items():
            if not key.startswith("__"):  # skip built-ins
                setattr(self, key, value)

        # save globals as attribute
        self.config_dict = combined_namespace
        


    def _init_cell(self, row, col):
        # init the morphology of the tissue
        is_neuron = (self.neurons_out and 
                     neuron_initiation.outline(self.num_layers ,self.num_frames, row, self.num_rows, col, self.num_cols))
        
        return is_neuron
    
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

    def plot_tissue(self, color_by, velocity_field=False, ax=None, legend=False):
        # get positions
        pos = nx.get_node_attributes(self.graph, 'pos')

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            created_fig = True
        x_length = self.num_cols * 1.5 * self.config_dict['cell_radius']
        y_length = self.num_rows * 1.5 * self.config_dict['cell_radius']
        ax.set_xlim(-0.05 * x_length, (1+0.05) * x_length)
        ax.set_ylim(-0.05 * y_length, (1+0.2) * y_length)    

        # Draw edges grouped by type - anything but non-neuron + marginal edges
        edge_colors = {
            'marginal': ('black',0.7),
            'internal': ('grey', 0.3),
            'boundary': ('green', 0.4),
        }
        
        for edge_type, (color, width) in edge_colors.items():
            edge_list = [
                (u, v) for u, v, d in self.graph.edges(data=True)
                if (d.get('edge_type') == edge_type)
            ]
            edge_collection  = nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=edge_list,
                edge_color=color,
                width=width,
                ax=ax
            )
            # Set zorder manually
            if edge_collection is not None:
                if isinstance(edge_collection, list):
                    for ec in edge_collection:
                        ec.set_zorder(5)
                else:
                    edge_collection.set_zorder(5)
                    
        node_colors = [
            'green' if self.graph.nodes[node].get("boundary") else 'gray'
            for node in self.graph.nodes
        ]
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=2,
            node_color=node_colors,
            ax=ax
        )

         #draw a dashed red outline around the inner frame regions
        self.draw_frame_outlines(ax)

        # Draw velocity field, if requested
        if velocity_field:
            for node in self.graph.nodes:
                x, y = self.graph.nodes[node]['pos']
                vx, vy = self.graph.nodes[node]['velocity']
                ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', color='yellow', scale=0.8, zorder=15, width = 0.003)

         
        # Dynamically get cell attribute method
        if color_by == 'height':
            self._coloring_by_value(self.initial_height, lambda cell: cell.get_height(), ax)
        elif color_by == 'area':
            self._coloring_by_value(self.initial_area, lambda cell: cell.get_area(), ax)
        else:
            raise ValueError(f"Unknown color_by value: {color_by}")

        if legend:
            if self.neurons_out:
                label_outside = "outside: neurons"
                label_window = "window: non-neurons"
            else:
                label_outside = "outside: non-neurons"
                label_window = "window: neurons"

            legend_elements = [
                Line2D([0], [0], linestyle='None', marker='', label=label_outside),
                Line2D([0], [0], linestyle='None', marker='', label=label_window)
            ]

            legend = ax.legend(handles=legend_elements, loc='upper right', framealpha=1)
            legend.set_zorder(1000)

        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()

        if created_fig:
            plt.show()

    def _coloring_by_value(self, center_value, get_func, ax, area=True, colorbar=True, factor=2):
        min_val = np.max(center_value / factor, 0)
        max_val = center_value * factor

        # Use TwoSlopeNorm to center the colormap on center_value
        norm = colors.TwoSlopeNorm(vmin=min_val, vcenter=center_value, vmax=max_val)

        base_cmap = cm.get_cmap(self.base_cmap)
        # Create cropped version to compress bright center
        darker_cmap = LinearSegmentedColormap.from_list(
            'darker_coolwarm',
            base_cmap(np.linspace(0.2, 0.8, 256))
        )
        
        for cell in self.cells:
            node_keys = cell.get_nodes()
            node_positions = [self.graph.nodes[key]['pos'] for key in node_keys]
            
            #color_map = self.inner_border_cmap if cell.inner_border[0] else darker_cmap
            color_map = darker_cmap

            val = get_func(cell)
            val = max(val, 1e-6)
            stretched_val = center_value + np.sign(val - center_value) * abs(val - center_value)**2.0  # or 2.5


            color = cm.get_cmap(color_map)(norm(stretched_val))
            ax.fill(*zip(*node_positions), color=color, alpha=0.8)

        # add colorbar
        if colorbar:
            param = 'Area' if area else 'Height'
            sm = cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, location='left')
            cbar.set_label(f'{param}', rotation=90, labelpad=10)
            cbar.ax.tick_params(labelsize=8)




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
        ax.set_xlim(self.min_height, self.max_height)

        if log_scale:
            ax.set_xscale('log')
            self.min_height_log, self.max_height_log = np.log10(self.min_height), np.log10(self.max_height)
            ax.set_xlim(self.min_height_log, self.max_height_log)
            ax.set_title('Distribution of Cell Heights (Log Scale)')

        plt.show()


    def compute_all_velocities(self):
        for node in self.graph.nodes:
            fx, fy = np.array(self.graph.nodes[node]['force'], dtype=float)

            # compute velocity
            velocity = self.mu * fx, self.mu * fy
            self.graph.nodes[node]['velocity'] = velocity
    
    def _compute_force(self, force_name: str, v1, v2):
        """
        Compute the force between two vertices using the specified force name.
        The force name should match the method name in the class.
        """
        force_method = getattr(self, f"_f_{force_name}")
        return force_method(v1, v2)

    def zeroing_forces(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]['force'] = np.array([0.0,0.0] ,dtype=float)
    
    def draw_frame_outlines(self, ax):
        """
        Draw dashed red rounded rectangles around inner frame regions.
        Geometry is calculated once and reused.
        """
        if not hasattr(self, 'frame_bounds'):
            self.frame_bounds = neuron_initiation.get_all_frame_bounds(
                num_frames=self.num_frames,
                num_cols=self.num_cols,
                num_rows=self.num_rows,
                num_layers=self.num_layers
            )

        dx = 3 / 2 * self.cell_radius
        dy = math.sqrt(3) * self.cell_radius

        for x_start, x_end, y_start, y_end in self.frame_bounds:
            patch = FancyBboxPatch(
                (x_start * dx - self.cell_radius, y_start * dy),  # corrected for hex grid offset
                (x_end - x_start) * dx,
                (y_end - y_start) * dy,
                boxstyle="round,pad=0.02,rounding_size=3",
                linewidth=2,
                linestyle='--',
                edgecolor='red',
                facecolor='none',
                zorder=50
            )
            ax.add_patch(patch)

    def compute_all_forces(self):
        """
        Compute the sum of forces acting on the vertices of the graph.
        """
        self.zeroing_forces()
        forces = list(self.forces)
        
        if "push_out" in forces:
            inner_border_cells = [cell for cell in self.cells if cell.inner_border[0] is True]
            for cell in inner_border_cells:
                self._f_push_out(cell)   
            forces.remove("push_out")
        
        #  iterate over each unique edge
        for v1, v2 in self.graph.edges:
                

                # extract forces
                force_v1 = self.graph.nodes[v1]['force']
                force_v2 = self.graph.nodes[v2]['force']
                for force_name in forces:
                    # compute force
                    temp_force_v1, temp_force_v2 = self._compute_force(force_name, v1, v2)  

                    # add forces
                    if not self.graph.nodes[v1].get("boundary"):
                        force_v1 += temp_force_v1
                    if not self.graph.nodes[v2].get("boundary"):
                        force_v2 += temp_force_v2
            
                self.graph.nodes[v1]['force'] = force_v1
                self.graph.nodes[v2]['force'] = force_v2

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
            velocity = self.mu * force

            # update position
            new_pos = pos + velocity * dt
            self.graph.nodes[node]['pos'] = tuple(new_pos)

        # update time step
        self.time_step += 1

    def _f_spring(self, v1, v2):
        """
        Calculate the spring force between two vertices.
        """
        dx, dy, dist = self._get_distances(v1,v2)
        # Avoid division by zero
        if dist < 1e-3:
            print(f"distance of nodes: {v1}, {v2} is almost zero!")
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
        # get spring constant accorfing to edge type
        edge_type = self._get_edge_type(v1, v2)
        spring_constant = self._get_spring_constant(v1, v2, edge_type)
        min_length = getattr(self,f"{edge_type}_min_length")
        rest_length = self._get_rest_length(v1, v2, edge_type)
        force_magnitude = spring_constant * (dist - rest_length)
        force_vector = np.array([force_magnitude * dx / dist, force_magnitude * dy / dist])
    
        return force_vector, (force_vector *(-1))
    
    def _get_spring_constant(self, v1, v2, edge_type):
        if self._is_nueron_edge(v1, v2):
            return getattr(self,f"neuron_spring_constant_{edge_type}")
        else:
            return getattr(self,f"non_neuron_spring_constant_{edge_type}")


    def _get_rest_length(self, v1, v2, edge_type):
        if self._is_nueron_edge(v1, v2):
            rest_length = getattr(self,f"{edge_type}_rest_length")
        else:
            rest_length = getattr(self,f"non_neuron_{edge_type}_rest_length")
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
            force_vector = self.line_tension_constant * unit_vector

            return force_vector, (force_vector *(-1))
        else:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
    def _f_push_out(self, cell, tangent_indices = [2,4]):
        """
        Calculate the outward force, if:
            - v1 and v2 are not neurons
            - v1 and v2 are on the inner border of a frame

        returns the normalized force oposite to the direction to the center of the frame

        push out can be constant, decay linearly or exponentially
        linear: f = F * k ** t
        exponential: f = F * e^(k * t)
        where k is the decay constant and t is the time step.
        """
        border_type = cell.inner_border[1]

        nodes = cell.nodes

        tangent_node1 = nodes[tangent_indices[0]]
        tangent_node2 = nodes[tangent_indices[1]]

        pose1 = self.graph.nodes[tangent_node1]['pos']
        pose2 = self.graph.nodes[tangent_node2]['pos']

        # Tangent vector: from pose1 to pose2
        tangent = np.array(pose2) - np.array(pose1)

        # Normalize the tangent vector (optional but often useful)
        tangent = tangent / np.linalg.norm(tangent)

        # Perpendicular vector (rotate 90° counter-clockwise in 2D)
        if border_type == "right":
            # vector as usual (90° counter-clockwise)
            vector = np.array([-tangent[1], tangent[0]])
        elif border_type == "left":
            # Opposite of the usual vector (90° clockwise)
            vector = np.array([tangent[1], -tangent[0]])
        elif border_type == "bottom":
            # Rotate 90° clockwise from tangent
            vector = -np.array(tangent)
        elif border_type == "top":
            # Rotate 90° counter-clockwise from tangent
            vector = np.array(tangent)


        for v1 in nodes:
            v1_force = self.graph.nodes[v1]['force']

            # constant
            if self.push_out_decay_type == 'constant':
                push_force = vector * self.push_out_force_strength
            
            # linear
            elif self.push_out_decay_type == 'linear':
                decay_ratio = max(0, 1 - self.time_step * self.push_out_decay_constant_lin)
                push_force = vector * self.push_out_force_strength * decay_ratio

            # exponential
            elif self.push_out_decay_type == 'exp':
                push_force = vector * self.push_out_force_strength  * np.exp(-self.time_step * self.push_out_decay_constant_exp)


            else:
                raise ValueError(f"Unknown push out decay type: {self.push_out_decay_type}")
            
            # ensure the force is at least the minimum push out force
            if np.linalg.norm(push_force) < self.min_push_out_force:
                push_force = vector * self.min_push_out_force
            
            # apply the force to the vertex
            self.graph.nodes[v1]['force'] = (v1_force + push_force)


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

            cell.update_area(area)

            new_height = self.cell_volume / area  
            cell.update_height(new_height)
    

    def compute_total_area(self, window_only=True):
        """
        Compute the total area of the tissue.
        """
        total_area = 0
        for cell in self.cells:
            # Determine if the cell should be excluded based on neuron status and neurons_out flag
            exclude_cell = (self.neurons_out and cell.is_neuron()) or (not self.neurons_out and not cell.is_neuron())

            # keep only cells insie the window 
            if window_only and exclude_cell:
                continue

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
            return round(val) if abs(val - round(val)) < 1e-3  else round(val, 3)
    
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
            spring_constant = self._get_spring_constant(v1, v2, edge_type)
            rest_length = self._get_rest_length(v1, v2, edge_type)

            # Line tension energy: 0.5 * k * (delta_x)^2
            if self._is_nueron_edge(v1, v2):
                line_tension_energy = 0.5 * self.line_tension_constant * (dist)**2
                total_energy += line_tension_energy
            
            # Spring energy: 0.5 * k * (l - l0)^2
            spring_energy = 0.5 * spring_constant * (dist - rest_length)**2
            total_energy += spring_energy
        return total_energy
    
    def _define_vertical_window(self,  num_vertical_rows=0.2):
        if self.num_vertical_rows:
            num_vertical_rows = self.num_vertical_rows
        
        if num_vertical_rows != 1.0:
            # Compute number of rows to include
            num_rows_in_window = int(self.num_rows * num_vertical_rows)
            row_mid = self.num_rows // 2
            row_lower = row_mid - (num_rows_in_window // 2)
            row_upper = row_mid + (num_rows_in_window + 1) // 2 - 1  # inclusive upper bound

            # Clamp to valid range
            row_lower = max(0, row_lower)
            row_upper = min(self.num_rows - 1, row_upper)
            # num_vertical_rows = int(self.num_rows * num_vertical_rows)
            # total_height = y_max - y_min
            # row_height = total_height / self.num_rows
            # window_height = num_vertical_rows * row_height
            # y_mid = (y_min + y_max) / 2
            # y_lower = y_mid - window_height / 2
            # y_upper = y_mid + window_height / 2
        else:
            row_lower, row_upper = 0, (self.num_rows - 1)

        return row_lower, row_upper
        

    def calculate_velocity_profile(self, bin_length=None):
        """
        Calculate the velocity profile of the tissue.
        Divide the x-axis into bins and compute the average velocity in each bin.
        If num_rows is specified, only include nodes within the middle num_rows rows.
        If it is None, use 0.2 of the total rows, if it is 'all', include all rows.
        """
        x_velocities, x_positions = [], []
        
        
        row_lower, row_upper = self._define_vertical_window()

        # Get x coords and velocities for nodes in middle rows
        for node in self.graph.nodes:
            temp_row = self.graph.nodes[node]['row']
            if row_lower <= temp_row <= row_upper:
                vx = self.mu * self.graph.nodes[node]['force'][0]
                x_velocities.append(vx)
                x_positions.append(self.graph.nodes[node]['pos'][0]) # add x positions 

        # Sort
        sorted_indices = np.argsort(x_positions)
        x_positions = np.array(x_positions)[sorted_indices]
        x_velocities = np.array(x_velocities)[sorted_indices]

        if bin_length is None:
            return x_positions, x_velocities

        bins = np.arange(min(x_positions), max(x_positions), bin_length)
        bin_indices = np.digitize(x_positions, bins) - 1
        valid = (bin_indices >= 0) & (bin_indices < len(bins) - 1)
        bin_indices = bin_indices[valid]
        velocities = x_velocities[valid]

        sum_per_bin = np.bincount(bin_indices, weights=velocities, minlength=len(bins)-1)
        count_per_bin = np.bincount(bin_indices, minlength=len(bins)-1)
        bin_means = np.divide(sum_per_bin, count_per_bin, out=np.zeros_like(sum_per_bin), where=count_per_bin != 0)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        return bin_centers, bin_means
    
    def compute_inner_border_x_velocity_middle_band(self):
        """
        Compute the average x-axis velocity of inner-border cells 
        that lie within the vertical window (y-axis).
        """
        lower_bound, upper_bound = self._define_vertical_window()


        x_velocities = []

        for cell in self.cells:
            is_border, _ = cell.inner_border
            if is_border:
                # Get row index from any node in the cell
                node_keys = cell.get_nodes()
                rows = [self.graph.nodes[node]['row'] for node in node_keys]
                cell_mean_row_loc = sum(rows) / len(rows)

                if lower_bound <= cell_mean_row_loc <= upper_bound:
                    # Get average vx across all cell nodes
                    vx = np.mean([abs(self.graph.nodes[n]['velocity'][0]) for n in node_keys])
                    x_velocities.append(vx)

        if x_velocities:
            return np.mean(x_velocities)
        else:
            return np.nan


