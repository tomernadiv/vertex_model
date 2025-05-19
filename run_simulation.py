
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



def run_simulation(T:tissue.Tissue, time_limit:int, output_dir, 
                    save_frame_interval = 10, dt=0.0001, 
                    dispaly_forces:bool = False,
                    forces=['spring'],
                    velocity_field=False):

    
    total_energy = []
    all_area_perc = []

    #init energy
    initial_energy = T.compute_total_energy()
    total_energy.append(initial_energy)

    #init area
    initial_total_area = T.compute_total_area()
    all_area_perc.append(100)

    # create outdir
    output_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    # iterate overthe graph
    for t in range(0, time_limit):
        print(f"\n---------------------Time {t}---------------------")

        if t % save_frame_interval == 0:

            # Save the tisuue frame
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Side-by-side axes
            ax2.set_title(f"Timestamp: {t}")
            T.compute_all_velocities()
            T.plot_tissue(color_by = 'area', ax=ax2, legend=True, velocity_field=velocity_field)

            # Energy plot
            ax1.plot(range(t + 1), total_energy, color='tab:red')
            ax1.set_title("Total Energy Over Time")
            ax1.set_xlabel("Time step")
            ax1.set_ylabel("Total Energy")
            ax1.set_xlim(0, time_limit)
            ax1.set_ylim(0, initial_energy*1.5)
            ax1.grid(True)

            # Area plot
            ax3.plot(range(t + 1), all_area_perc, color='tab:blue')
            ax3.set_title("Area Percentage Over Time")
            ax3.set_xlabel("Time step")
            ax3.set_ylabel("% Area")
            ax3.set_xlim(0, time_limit)
            ax3.set_ylim(0, 150)
            ax3.grid(True)

            # Add text

            # total energy
            ax2.text(0.0, 0.025, f"Energy: {total_energy[-1]:.3f}", 
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax2.transAxes,
                    fontsize=15, bbox=dict(facecolor='red', alpha=0.5, boxstyle='round'))
            

            # precentage of the area
            ax2.text(0.85, 0.025, f"Area: {all_area_perc[-1]:.3f}%", 
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax2.transAxes,
                    fontsize=15, bbox=dict(facecolor='blue', alpha=0.5, boxstyle='round')),


            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"tissue_frame_{t}.png"))
            plt.close(fig)
        
        # Update for next iteration
        T.compute_all_forces(forces)
        T.update_positions(dt=dt)
        T.update_heights()

        total_energy.append(T.compute_total_energy())
        temp_total_area = T.compute_total_area()
        area_perc = (temp_total_area/initial_total_area) * 100
        all_area_perc.append(area_perc)

        if dispaly_forces:
            dispaly_forces_func(T)
    
    return total_energy


def replay_simulation(frames_dir, simulation_name: str, fps: int = 2):

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
    plt.tight_layout()
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




def simulation(tissue_size, time_limit, save_frame_interval, dt, num_out_layers, n_init_func, simulation_name, forces, pertubation=False, velocity_field=False):

    # sanity check orints
    print(f"Simulation Parameters: {simulation_name}")
    print(f"dt: {dt}")
    print(f"tissue_size: {tissue_size}")
    print(f"time_limit: {time_limit}")
    print(f"save_frame_interval: {save_frame_interval}")


    T = tissue.Tissue(cell_radius=cell_radius, num_cols=tissue_size, num_rows=tissue_size, n_init_func=n_init_func, num_out_layers=num_out_layers)

    if pertubation:
        # add pertubation
        add_pertubation(T)
    

    # check if saving folder exists, if it is, erase it
    output_dir = os.path.join('results', simulation_name)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # redirect stdout to log file
    original_stdout = sys.stdout
    sys.stdout = open(os.path.join(output_dir, "log.txt"), "w")

    total_energy = run_simulation(T=T, time_limit=time_limit, forces=forces, output_dir=output_dir, dt=dt, save_frame_interval=save_frame_interval, velocity_field=velocity_field)
    replay_simulation(frames_dir=os.path.join(output_dir, "frames"), simulation_name=simulation_name)
    plot_energy_graph(simulation_name=simulation_name, total_energy=total_energy, save_graph=True)
    np.save(os.path.join('results', simulation_name, "energy.npy"), total_energy)

    sys.stdout = original_stdout
    print("Done.")



if __name__ == "__main__":
    run_name='line_tension_strong_w_velocity_field'
    forces = ['spring', 'line_tension']
    tissue_size = 12
    time_limit = 500
    save_frame_interval = 5
    dt = 0.1
    num_out_layers = 2
    n_init_func = "all_neurons"
    simulation_name = f"{run_name}_size{tissue_size}_lim{time_limit}_dt{dt}_{n_init_func}_shrink{shrinking_const}"

    simulation(tissue_size, time_limit, save_frame_interval, dt, num_out_layers, n_init_func, simulation_name, forces=forces, pertubation=False, velocity_field=True)


    