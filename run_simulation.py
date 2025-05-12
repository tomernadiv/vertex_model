
from globals import *
import tissue
import pickle



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

def dispaly_forces_func(T):
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



def run_simulation(simulation_name:str, T:tissue.Tissue, time_limit:int,
                    save_frame_interval = 10, dt=0.0001, 
                    dispaly_forces:bool = False):
    
    # check if saving folder exists, if it is, erase it
    output_dir = os.path.join('results', simulation_name)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    total_energy = []
    initial_energy = T.compute_total_energy()
    total_energy.append(initial_energy)


    initial_total_area = T.compute_total_area()
    output_dir = os.path.join('results', simulation_name, "frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate overthe graph
    for t in range(0, time_limit):
        print(f"\n---------------------Time {t}---------------------")


        if t % save_frame_interval == 0:

            # Save the tisuue frame
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Side-by-side axes
            ax1.set_title(f"Timestamp: {t}")
            T.plot_tissue(ax=ax1) 

            # Energy plot
            ax2.plot(range(t + 1), total_energy, color='tab:red')
            ax2.set_title("Total Energy Over Time")
            ax2.set_xlabel("Time step")
            ax2.set_ylabel("Total Energy")
            ax2.set_xlim(0, time_limit)
            ax2.set_ylim(0, initial_energy*5)
            ax2.grid(True)

            # precentage of the area
            temp_total_area = T.compute_total_area()
            area_perc = (temp_total_area/initial_total_area) * 100
            ax1.text(0.95, 0.025, f"Area: {area_perc:.1f}%", 
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax1.transAxes,
                    fontsize=15, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"tissue_frame_{t}.png"))
            plt.close(fig)
        
        # Update for next iteration
        T.compute_all_forces(['spring'])
        T.update_positions(dt=dt)
        T.update_heights()
        total_energy.append(T.compute_total_energy())

        if dispaly_forces:
            dispaly_forces_func(T)
    
    return total_energy


def replay_simulation(simulation_name: str, fps: int = 2):
    frames_dir = os.path.join('results', simulation_name, "frames")
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Could not find frames directory at {frames_dir}")
    
    # Get and sort all frame paths
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # assumes "tissue_frame_#.png"
    )
    if not frame_files:
        raise FileNotFoundError("No PNG frames found in frames directory.")

    fig, ax = plt.subplots()
    img = ax.imshow(mpimg.imread(os.path.join(frames_dir, frame_files[0])))
    ax.axis('off')  # optional: hide axes

    def update(i):
        img.set_data(mpimg.imread(os.path.join(frames_dir, frame_files[i])))
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_files), interval=1000/fps, blit=True, repeat=False
    )

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


def simulation(dt, tissue_size, save_frame_interval, time_limit):
    
    simulation_name = f'test_dt={dt}'

    # sanity check orints
    print("Simulation Parameters:")
    print(f"dt: {dt}")
    print(f"tissue_size: {tissue_size}")
    print(f"time_limit: {time_limit}")
    print(f"save_frame_interval: {save_frame_interval}")

    # initialize tissue
    T = tissue.Tissue(cell_radius=cell_radius, num_cols=tissue_size, num_rows=tissue_size, n_init_func="all_neurons", num_out_layers=2)

    # create rsults directory
    results_dir = os.path.join('results', simulation_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # sys.stdout = open(os.path.join(results_dir, "log.txt"), "w")
    
    # run
    print(f"Starting Simulation for {time_limit} intervals:")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    total_energy, all_amplitudes, all_area_percs = run_simulation(simulation_name=simulation_name, T=T, time_limit=time_limit, save_frame_interval = save_frame_interval, dt=dt)
    print("\nFinished Simulation Succesfully.")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    
    # save plots
    replay_simulation(simulation_name=simulation_name)
    plot_energy_graph(simulation_name=simulation_name, total_energy=total_energy, save_graph=True)
    plt.close()
    return total_energy, all_amplitudes, all_area_percs







def convergence_plots():
    dts = [i for i in range(6,7)]
    max_time_limit = 10**6
    time_limits = [min(int(10**(i+2)),max_time_limit) for i in range(1,len(dts)+1)]
    tissue_size = 10
    results = {'amps': {},
               'energy': {}, 
               'area': {}}
    

    print("Simulation Iterations:")
    for i in range(len(dts)):
        dt = 10 ** -dts[i]
        time_limit = time_limits[i]
        print(f"dt: {dt}, time_limit: {time_limit}")

    # iterate on all dt values
    for i in range(len(dts)):
        dt = 10 ** -dts[i]
        time_limit = 10**6
        save_frame_interval = int(time_limit // 100) # save 100 frames
    
        # run simulation
        energy_per_frame, all_amplitdes, all_area_percs = simulation(dt, tissue_size, save_frame_interval, time_limit)
        results['amps'][i+1] = all_amplitdes
        results['energy'][i+1] = energy_per_frame
        results['area'][i+1] = all_area_percs

    # save the results dict using pickle
    with open('results2.pkl', 'wb') as f:
        pickle.dump(results, f)

    # plot average convergence amplitude as function of dt
    means = [np.mean(results['amps'][i]) for i in dts]
    stds = [np.std(results['amps'][i]) for i in dts]
    dts = [10 ** -i for i in dts]
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(dts, means, 'o-')
    plt.fill_between(dts, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    ax.set_xscale('log')
    ax.set_xlabel("dt")
    ax.set_ylabel("Amplitude")
    ax.set_title("Average (and STD) of Energy Amplitude as Function of dt")
    ax.grid(True)
    plt.savefig('amplitude_plot2.png')


if __name__ == "__main__":
    simulation_name='one_hex'
    T = tissue.Tissue(cell_radius=cell_radius, num_cols=1, num_rows=1, n_init_func="all_neurons", num_out_layers=5)
    total_energy = run_simulation(simulation_name=simulation_name, time_limit=5000, T=T, dt=0.000001, save_frame_interval=100)
    replay_simulation(simulation_name=simulation_name)
    plot_energy_graph(simulation_name=simulation_name, total_energy=total_energy, save_graph=True)
    np.save(os.path.join('results', simulation_name, "energy.npy"), total_energy)
