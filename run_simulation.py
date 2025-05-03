
from globals import *
import tissue
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import os



# add some pertubations to check plotting
def add_pertubation(T:tissue.Tissue):
    original_pos = nx.get_node_attributes(T.graph, 'pos').copy()
    for cell in T.cells:
        cell.height = np.random.uniform(0.1, 1.0)
        pts = np.array([original_pos[n] for n in cell.nodes])
        center = pts.mean(axis=0)
        for i, n in enumerate(cell.nodes):
            x, y = original_pos[n]
            vx, vy = x - center[0], y - center[1]
            length = np.hypot(vx, vy) or 1.0
            ux, uy = vx/length, vy/length
            shift_x = np.random.normal(0, 0.1)
            shift_y = np.random.normal(0, 0.1)
            new_pos = (x + ux*shift_x, y + uy*shift_y)
            T.graph.nodes[n]['pos'] = new_pos


def run_simulation(T:tissue.Tissue, time_limit:int, output_dir:str):
    total_energy = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reset all forces to zero  
    for node in T.graph.nodes:
        T.graph.nodes[node]['force'] = np.array([0.0, 0.0])

    # iterate overthe graph
    for t in range(1, time_limit):
        # Create a new figure each time
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot tissue
        ax.set_title(f"Timestamp: {t}")
        T.plot_tissue(ax=ax)
        
        # Save the plot as an image (optional)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f"tissue_frame_{t}.png"))  # Or comment this out if not needed
        plt.close(fig)
        
        # Pause for visualization
        time.sleep(0.5)
        
        # Update for next iteration
        T.compute_all_forces(['spring'])  # or ['spring', 'line_tension'] if needed
        T.update_positions(dt=0.5)
        T.update_heights()
        total_energy.append(T.compute_total_energy())

        #dispaly forces on each node
        for i, node in enumerate(T.graph.nodes):
            cell_type = ""
            if T.graph.nodes[node]['neuron']:
                cell_type +="neuron"
            if T.graph.nodes[node]['boundary']:
                if cell_type:
                    cell_type += "_"
                cell_type += "boundary"
            print(f"Node {i}: {cell_type}  {node} {T.graph.nodes[node]['force']}")
    
    return total_energy


def replay_simulation(frames_dir:str, num_of_frames:int):
    fig, ax = plt.subplots()

    first_frame_path = os.path.join(frames_dir, "tissue_frame_1.png")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"Could not find first frame at {first_frame_path}")
    img = ax.imshow(mpimg.imread(first_frame_path))

    def update(frame):
        frame_path = os.path.join(frames_dir, f"tissue_frame_{frame}.png")
        img.set_data(mpimg.imread(frame_path))
        ax.set_title(f"Timestamp: {frame}")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=range(1, num_of_frames+1), interval=500, blit=True, repeat=True)

    plt.tight_layout()
    return ani, fig  

def plot_energy_graph(total_energy, save_graph:bool = False, output_path:str = None):
    
    plt.plot(total_energy)
    plt.xlabel("Time step")
    plt.ylabel("Total Energy")
    plt.title("Total Energy Over Time")
    plt.grid(True)

    if save_graph and (output_path != None):
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == "__main__":
    
    #init
    T = tissue.Tissue(cell_radius=cell_radius, num_cols=10, num_rows=10)
    time_limit=10
    frames_dir = "./frames_dir"

    # run
    #add_pertubation(T)
    total_energy = run_simulation(T, time_limit=time_limit, output_dir=frames_dir)

    # Show the animation
    ani, fig = replay_simulation(frames_dir=frames_dir, num_of_frames=time_limit)
    plt.rcParams['animation.html'] = 'html5'
    plt.figure(fig.number) 
    plt.show()


    plot_energy_graph(total_energy)



