
from configs.run_config import *
import tissue

# plotting function
def plot_timestamp(T: tissue, t: int, energy, area, position, velocity, time_limit,
                   output_dir=None, show_velocity_field: bool = False, show: bool = False):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1])

    # Big tissue plot on the left (spans both rows)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title(f"Timestamp: {t}")
    T.plot_tissue(color_by='area', ax=ax1, legend=True, velocity_field=show_velocity_field)

    # Velocity profile (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Velocity Profile")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("X Velocity")
    sns.scatterplot(x=position, y=velocity, ax=ax2, color='tab:green', s=10)
    sns.lineplot(x=position, y=velocity, ax=ax2, color='tab:green', linewidth=0.1)

    # Energy + Area % over time (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(range(t + 1), energy, color='tab:red')
    ax3.set_title("Total Energy and Area % Over Time")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Total Energy", color='tab:red')
    ax3.set_xlim(0, time_limit)
    ax3.set_ylim(0, energy[0] * 1.5)
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.grid(True)

    ax4 = ax3.twinx()
    ax4.plot(range(t + 1), area, color='tab:blue')
    ax4.set_ylabel("% Area", color='tab:blue')
    ax4.set_ylim(0, 150)
    ax4.set_xlim(0, time_limit)
    ax4.tick_params(axis='y', labelcolor='tab:blue')

    # Text annotations on the tissue plot
    ax1.text(0.02, 0.025, f"Energy: {energy[-1]:.3f}",
             transform=ax1.transAxes,
             fontsize=14, bbox=dict(facecolor='red', alpha=0.5, boxstyle='round'))

    ax1.text(0.75, 0.025, f"Area: {area[-1]:.3f}%",
             transform=ax1.transAxes,
             fontsize=14, bbox=dict(facecolor='blue', alpha=0.5, boxstyle='round'))

    plt.tight_layout()

    if show:
        plt.show()

    if output_dir:
        fig.savefig(os.path.join(output_dir, f"tissue_frame_{t}.png"))

    plt.close(fig)


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

def run_simulation(T:tissue.Tissue,                  # tissue object
                   time_limit:int,                   # number of timestamps
                   dt=0.1,                           # time step  
                   output_dir='',                    # output directory for saving frames
                   save_frame_interval = 10,         # save frame every x timestamps
                   forces=['spring'],                # forces to apply, e.g. ['spring', 'line_tension']
                   velocity_profile_position_bin=1,  # bin size for velocity profile
                   show_velocity_field=False):       # whether to show velocity field in the tisssue plot

    
    # init empty stacks
    energies = []
    area_percs = []
    velocity_profiles = []


    #init energy
    initial_energy = T.compute_total_energy()
    energies.append(initial_energy)

    #init area
    initial_total_area = T.compute_total_area()
    area_percs.append(100)

    # create outdir
    output_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    # iterate overthe graph
    for t in range(0, time_limit):
        print(f"\n---------------------Time {t}---------------------")

        # compute velocities
        T.compute_all_velocities()

        # compute velocity profile
        position, velocity = T.calculate_velocity_profile(bin_length=velocity_profile_position_bin)
        velocity_profiles.append((position, velocity))

        # plot timestamp
        if t % save_frame_interval == 0:
            plot_timestamp(T=T, t=t, energy=energies,
                           area=area_percs, position=position,
                           velocity=velocity, time_limit=time_limit,
                           output_dir=output_dir,
                           show_velocity_field=show_velocity_field)
        # Update for next iteration
        T.compute_all_forces(forces)
        T.update_positions(dt=dt)
        T.update_heights()

        # add to stacjs
        energies.append(T.compute_total_energy())
        area_percs.append((T.compute_total_area()/initial_total_area) * 100)

    # stack up results
    positions = [np.array(vp[0]) for vp in velocity_profiles]
    velocities = [np.array(vp[1]) for vp in velocity_profiles]
    res = {'energy': np.array(energies),
           'areaL': np.array(area_percs),
           'velocity_profile_position': positions,
           'velocity_profile_velocity': velocities}
    
    # save results
    with open(os.path.join(output_dir, "..", "results.pkl"), 'wb') as f:
        pickle.dump(res, f)
    
    return res

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




def simulation(time_limit, save_frame_interval, dt, globals_config_path, simulation_config_path, morphology_config_path, simulation_name, forces, pertubation=False, show_velocity_field=False, rm_frames = True):

    # sanity check orints
    print(f"Simulation Parameters: {simulation_name}")
    print(f"dt: {dt}")
    print(f"time_limit: {time_limit}")
    print(f"save_frame_interval: {save_frame_interval}")


    T = tissue.Tissue(globals_config_path = globals_config_path, simulation_config_path = simulation_config_path, morphology_config_path = morphology_config_path)


    # add pertubations if needed
    if pertubation:
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

    # run simulation
    result_dict = run_simulation(T=T, time_limit=time_limit, forces=forces,velocity_profile_position_bin=velocity_profile_position_bin, output_dir=output_dir, dt=dt, save_frame_interval=save_frame_interval, show_velocity_field=show_velocity_field)
    
    # create video
    replay_simulation(frames_dir=os.path.join(output_dir, "frames"), simulation_name=simulation_name)
    
    # move the last figure create to simulation dir and remove frame dir if specified
    shutil.copyfile(os.path.join(output_dir, "frames", f"tissue_frame_{(time_limit-save_frame_interval)}.png"), os.path.join(output_dir,"last_frame.png"))

    if rm_frames:
        shutil.rmtree(os.path.join(output_dir, "frames"))
    
    # redirect stdout back to console
    sys.stdout = original_stdout
    print("Done.")



if __name__ == "__main__":
    simulation_to_forces = {
        "simulation_1": ['spring', 'line_tension'],
        "simulation_2": ['spring', 'line_tension', 'push_out'],
        "simulation_3": ['spring', 'line_tension']
    }
    time_limit = 50
    save_frame_interval = 5
    dt = 0.1
    velocity_profile_position_bin = 0.5
    simulation_number = 1
    simulation_name = f"simulation_{simulation_number}"
    forces = simulation_to_forces[simulation_name]
    globals_config_path = "configs/globals.py"
    simulation_config_path = f"configs/{simulation_name}.py"
    morphology_config_path = "configs/morphology_config.py"

    simulation(time_limit, save_frame_interval, dt, globals_config_path, simulation_config_path, morphology_config_path, simulation_name, forces=forces, pertubation=False, show_velocity_field=True, rm_frames = True)



    