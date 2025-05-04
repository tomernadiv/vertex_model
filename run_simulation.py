
from globals import *
import tissue



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


def run_simulation(simulation_name:str, T:tissue.Tissue, time_limit:int):
    total_energy = []

    output_dir = os.path.join('results', simulation_name, "frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reset all forces to zero  
    for node in T.graph.nodes:
        T.graph.nodes[node]['force'] = np.array([0.0, 0.0])

    # iterate overthe graph
    for t in range(1, time_limit):
        print(f"\n---------------------Time {t}---------------------")
        # Create a new figure each time
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot tissue
        ax.set_title(f"Timestamp: {t}")
        T.plot_tissue(ax=ax)
        
        # Save the plot as an image (optional)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f"tissue_frame_{t}.png")) 
        plt.close(fig)
        
        # Pause for visualization
        time.sleep(0.5)
        
        # Update for next iteration
        T.compute_all_forces(['spring'])  # or ['spring', 'line_tension'] if needed
        T.update_positions(dt=0.5)
        T.update_heights()
        total_energy.append(T.compute_total_energy())

        # #dispaly forces on each node
        # for i, node in enumerate(T.graph.nodes):
        #     cell_type = ""
        #     if T.graph.nodes[node]['neuron']:
        #         cell_type +="neuron"
        #     if T.graph.nodes[node]['boundary']:
        #         if cell_type:
        #             cell_type += "_"
        #         cell_type += "boundary"
        #     print(f"Node {i}: {cell_type}  {node} {T.graph.nodes[node]['force']}")
    
    return total_energy

def replay_simulation(simulation_name, num_of_frames: int, fps: int = 2):
    frames_dir = os.path.join('results', simulation_name, "frames")
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Could not find frames directory at {frames_dir}")
    

    fig, ax = plt.subplots()

    first_frame_path = os.path.join(frames_dir, "tissue_frame_1.png")
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"Could not find first frame at {first_frame_path}")
    
    img = ax.imshow(mpimg.imread(first_frame_path))

    def update(frame):
        frame_path = os.path.join(frames_dir, f"tissue_frame_{frame}.png")
        img.set_data(mpimg.imread(frame_path))
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=range(1, num_of_frames), interval=500, blit=True, repeat=False)

    plt.tight_layout()

    # Save animation as MP4 using ffmpeg writer
    output_file = os.path.join('results', simulation_name, "simulation.mp4")
    ani.save(output_file, writer='ffmpeg', fps=fps)

    plt.close(fig)

def plot_energy_graph(simulation_name, total_energy, save_graph:bool = False):
    
    plt.plot(total_energy)
    plt.xlabel("Time step")
    plt.ylabel("Total Energy")
    plt.title("Total Energy Over Time")
    plt.grid(True)
    output_path = os.path.join('results', simulation_name, "energy_graph.png")

    if save_graph and (output_path != None):
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == "__main__":
    
    #init
    tissue_size = 5
    time_limit=20
    simulation_name = 'test_simulation'
    results_dir = os.path.join('results', simulation_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    sys.stdout = open(os.path.join(results_dir, "log.txt"), "w")
    T = tissue.Tissue(cell_radius=cell_radius, num_cols=tissue_size, num_rows=tissue_size)

    # run
    #add_pertubation(T)
    print(f"Starting Simulation for {time_limit} intervals:")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    frames_dir = os.path.join('results', simulation_name, "frames")
    total_energy = run_simulation(simulation_name=simulation_name, T=T, time_limit=time_limit)
    print("\nFinished Simulation Succesfully.")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # save plots
    replay_simulation(simulation_name, num_of_frames=time_limit)
    plot_energy_graph(total_energy, save_graph=True, output_path=os.path.join(results_dir,"energy_graph.png"))

    print(f"Saved Simulation on {results_dir}.")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Done.")



