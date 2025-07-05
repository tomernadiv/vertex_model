
from configs.run_config import *
import tissue
from configs.imports import *
from neuron_initiation import get_frame_bounds_from_index
import json

# plotting function
def plot_timestamp(T: tissue, t: int, title, energy, total_area, window_area, position, velocity, time_limit,
                   output_dir=None, show_velocity_field: bool = False, show: bool = False, max_velocity: float = None):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1])

    # Big tissue plot on the left (spans both rows)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title(f"Timestamp: {t}\n\nDescription: {title}")
    T.plot_tissue(color_by='area', ax=ax1, legend=True, velocity_field=show_velocity_field)

    # Velocity profile (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_velocity_profile(ax2, position, velocity, x_lim=T.num_cols * 1.5 * T.config_dict['cell_radius'] + 1, y_lim=max_velocity)


    # add 2 light gray rectangles to indicate the area of the windows
    total_width = T.num_cols * 1.5 * T.config_dict['cell_radius']
    marginal_width = T.config_dict['num_layers'] * 1.5 * T.config_dict['cell_radius']
    window_width = (total_width / 2) - (2 * marginal_width)
    for frame_index in range(T.num_frames):
        x_start, _ = get_frame_bounds_from_index(frame_index, T.num_cols, T.num_frames)
        window_start = x_start * 1.5 * T.config_dict['cell_radius']
        ax2.add_patch(patches.Rectangle((window_start + marginal_width, -3),
                                          window_width, 
                                          10, 
                                          color='lightgray', 
                                          alpha=0.5))

    # Energy + Area % over time (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(range(t + 1), energy, color='tab:red')
    ax3.set_title("Total Energy and Area % Over Time")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Total Energy", color='tab:red')
    ax3.set_xlim(0, time_limit)
    ax3.set_ylim(energy[0] * 0.3, energy[0] * 1.1)
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.grid(True)

    ax4 = ax3.twinx()
    ax4.plot(range(t + 1), window_area, color='tab:blue')
    ax4.set_ylabel("% Area", color='tab:blue')
    ax4.set_ylim(window_area[0] * 0.3, window_area[0]*3) 
    ax4.set_xlim(0, time_limit)
    ax4.tick_params(axis='y', labelcolor='tab:blue')


    # Text annotations on the tissue plot
    ax1.text(0.02, -0.05, f"Energy: {energy[-1]:.3f}",
             transform=ax1.transAxes,
             fontsize=14, bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round', zorder=10))
    
    # ax1.text(0.4, -0.05, f"Total Area: {total_area:.1f}%",
    #          transform=ax1.transAxes,
    #          fontsize=14, bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round', zorder=10))

    ax1.text(0.5, -0.05, f"Window Area: {window_area[-1]:.3f}%",
             transform=ax1.transAxes,
             fontsize=14, bbox=dict(facecolor='yellow', alpha=0.9, boxstyle='round', zorder=10))

    plt.tight_layout()

    if show:
        plt.show()

    if output_dir:
        fig.savefig(os.path.join(output_dir, f"tissue_frame_{t}.png"))

    plt.close(fig)

def plot_velocity_profile(ax, position, velocity, x_lim = None, y_lim = None):
    ax.set_title("Velocity Profile")
    ax.set_xlabel("X Position")
    ax.set_ylabel("X Velocity")
    sns.scatterplot(x=position, y=velocity, ax=ax, color='tab:green', s=10)
    sns.lineplot(x=position, y=velocity, ax=ax, color='tab:green', linewidth=0.1)

    if x_lim is not None:
        ax.set_xlim(0, x_lim)

    if y_lim is not None:
        ax.set_ylim(-1 * y_lim, y_lim)
    else:
        ax.set_ylim(-3, 3)
    


def plot_vx_border(simulation_name: str, vx_series: list[float]):
    """
    Plot mean absolute x-velocity of the inner border layer over time and save to file.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(vx_series, color='tab:orange')
    plt.xlabel("Time step")
    plt.ylabel("Mean |vx| (Inner Border)")
    plt.title("Mean X-Velocity of Inner Border Over Time")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join('results', simulation_name, "vx_inner_border_plot.png")
    plt.savefig(output_path)
    plt.close()

def plot_vx_vs_time(simulation_name: str, vx_series: list[float], y_label:str = "X Axis Velocity"):
    """
    Plot x-velocity over time and save to file.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(vx_series, color='tab:orange')
    plt.xlabel("Time step")
    plt.ylabel(y_label)
    plt.title(f"{y_label} Over Time")
    plt.grid(True)
    plt.tight_layout()

    fig_label = y_label.replace(" ", "_")
    output_path = os.path.join('results', simulation_name, f"{fig_label}_plot.png")
    plt.savefig(output_path)
    plt.close()

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
                   title:str,
                   dt=0.1,                           # time step  
                   output_dir='',                    # output directory for saving frames
                   save_frame_interval = 10,         # save frame every x timestamps
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
    initial_total_area = T.compute_total_area(window_only=False)
    initial_window_area = T.compute_total_area(window_only=True)
    area_percs.append(100)

    # create outdir
    output_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    #initialize a list of x velocities of the inner border layer
    vx_border_layer_series = []

    max_velocities = []

    # iterate overthe graph
    for t in range(0, time_limit):
        print(f"\n---------------------Time {t}---------------------")

        # compute velocities
        T.compute_all_velocities()

        # compute velocity profile
        position, velocity = T.calculate_velocity_profile(bin_length=velocity_profile_position_bin)
        velocity_profiles.append((position, velocity))
        max_velocities.append(max(abs(velocity)))
        # calculate mean velocity of inner bound layer
        vx_border = T.compute_inner_border_x_velocity_middle_band()
        vx_border_layer_series.append(vx_border)

        

        # plot timestamp
        if t % save_frame_interval == 0:
            plot_timestamp(T=T, t=t, title=title, energy=energies, total_area = (T.compute_total_area(window_only=False)/initial_total_area) * 100,
                           window_area=area_percs, position=position,
                           velocity=velocity, time_limit=time_limit,
                           output_dir=output_dir,
                           show_velocity_field=show_velocity_field)
        # Update for next iteration
        T.compute_all_forces()
        T.update_positions(dt=dt)
        T.update_heights()

        # add to stacjs
        energies.append(T.compute_total_energy())
        area_percs.append((T.compute_total_area()/initial_window_area) * 100)

    # stack up results
    positions = [np.array(vp[0]) for vp in velocity_profiles]
    velocities = [np.array(vp[1]) for vp in velocity_profiles]
    vx_inner_border = np.array(vx_border_layer_series)
    res = {'energy': np.array(energies),
           'areaL': np.array(area_percs),
           'velocity_profile_position': positions,
           'velocity_profile_velocity': velocities,
           'vx_inner_border' : vx_inner_border,
           'max_velocities': max_velocities
           }

    
    # save results
    with open(os.path.join(output_dir, "..", "results.pkl"), 'wb') as f:
        pickle.dump(res, f)

    return res

def run_only_for_window_size(T:tissue.Tissue,                  # tissue objects
                   time_limit:int,                   # number of timestamps
                   title:str,
                   dt=0.1,                           # time step  
                   output_dir='',                    # output directory for saving frames
                   save_frame_interval = 10,         # save frame every x timestamps
                   velocity_profile_position_bin=1,  # bin size for velocity profile
                   show_velocity_field=False):       # whether to show velocity field in the tisssue plot

    
    #init area
    initial_window_area = T.compute_total_area(window_only=True)

    # create outdir
    output_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    max_at_10 = 0
    # iterate overthe graph
    for t in range(0, time_limit+1):
        print(f"\n---------------------Time {t}---------------------")

        if t == 10:
            # compute velocities
            T.compute_all_velocities()
            # compute velocity profile
            _ , velocity = T.calculate_velocity_profile(bin_length=velocity_profile_position_bin)
            max_at_10 = max(abs(velocity))   

        # Update for next iteration
        T.compute_all_forces()
        T.update_positions(dt=dt)
        T.update_heights()

    # compute final area
    final_window_area = (T.compute_total_area(window_only=True)/initial_window_area) * 100

    return final_window_area, max_at_10

def replay_simulation(frames_dir, simulation_name: str, fps: int = 4):

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

def simulation(time_limit, save_frame_interval, dt, globals_config_path, simulation_config_path, morphology_config_path, simulation_name, velocity_profile_position_bin, pertubation=False, show_velocity_field=False, rm_frames = True):

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
    log_file = open(os.path.join(output_dir, "log.txt"), "w", buffering=save_frame_interval)
    sys.stdout = log_file

    # run simulation
    title = runpy.run_path(simulation_config_path)["description"]
    result_dict = run_simulation(T=T, time_limit=time_limit, title = title,  velocity_profile_position_bin=velocity_profile_position_bin, output_dir=output_dir, dt=dt, save_frame_interval=save_frame_interval, show_velocity_field=show_velocity_field)
    
    # create video
    replay_simulation(frames_dir=os.path.join(output_dir, "frames"), simulation_name=simulation_name)

    # plots:
    plot_vx_vs_time(simulation_name, vx_series=result_dict["vx_inner_border"], y_label="Inner Border Mean Abs vx")
    max_velocities = result_dict["max_velocities"]
    plot_vx_vs_time(simulation_name, vx_series=max_velocities, y_label="Max Abs vx")

    fig, ax = plt.subplots()
    max_time_point = np.argmax(max_velocities)
    max_profile_position = result_dict["velocity_profile_position"][max_time_point]
    max_profile_velocity = result_dict["velocity_profile_velocity"][max_time_point]
    plot_velocity_profile(ax, max_profile_position, max_profile_velocity, 
                          x_lim=T.num_cols * 1.5 * T.config_dict['cell_radius'] + 1, 
                          y_lim=np.max(max_profile_velocity)+0.5)
    plot_outpath = os.path.join(output_dir,"max_velocity_profile.png")
    fig.savefig(plot_outpath, dpi=300, bbox_inches='tight')
    
    
    # move the last figure create to simulation dir and remove frame dir if specified
    shutil.copyfile(os.path.join(output_dir, "frames", f"tissue_frame_{(time_limit-save_frame_interval)}.png"), os.path.join(output_dir,"last_frame.png"))

    if rm_frames:
        shutil.rmtree(os.path.join(output_dir, "frames"))
    
    # redirect stdout back to console
    log_file.flush()     # flush to disk
    log_file.close()     
    sys.stdout = original_stdout
    print("Done.")

def simulation_only_for_window_size(time_limit, save_frame_interval, dt, globals_config_path, simulation_config_path, morphology_config_path, simulation_name, velocity_profile_position_bin, pertubation=False, show_velocity_field=False, rm_frames = True):

    # sanity check orints
    print(f"Simulation Parameters: {simulation_name}")
    print(f"dt: {dt}")
    print(f"time_limit: {time_limit}")
    print(f"save_frame_interval: {save_frame_interval}")


    T = tissue.Tissue(globals_config_path = globals_config_path, simulation_config_path = simulation_config_path, morphology_config_path = morphology_config_path)
    initial_area = T.compute_total_area(window_only=False)

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
    log_file = open(os.path.join(output_dir, "log.txt"), "w", buffering=save_frame_interval)
    sys.stdout = log_file

    # run simulation
    title = runpy.run_path(simulation_config_path)["description"]
    last_window_size, max_at_10 = run_only_for_window_size(T=T, time_limit=time_limit, title = title,  velocity_profile_position_bin=velocity_profile_position_bin, output_dir=output_dir, dt=dt, save_frame_interval=save_frame_interval, show_velocity_field=show_velocity_field)
    final_area = T.compute_total_area(window_only=False)
    final_total_area = (final_area / initial_area) * 100


    simulation_consts = runpy.run_path(simulation_config_path)
    run_info = {
        "simulation_name": simulation_name,
        "time_limit": time_limit,
        "save_frame_interval": save_frame_interval,
        "dt": dt,
        "last_window_size": last_window_size,
        "max_at_10": max_at_10,
        "shrinking_const": simulation_consts["shrinking_const"],
        "expansion_const": simulation_consts["expansion_const"],
        "push_out_force_strength": simulation_consts["push_out_force_strength"],
        'final_total_area': final_total_area,
    }

    json_path = os.path.join(output_dir, "run_info.json")
    with open(json_path, 'w') as f:
        json.dump(run_info, f, indent=4)
    
    
    # redirect stdout back to console
    log_file.flush()     # flush to disk
    log_file.close()     
    sys.stdout = original_stdout
    print("Done.")




if __name__ == "__main__":

    if len(sys.argv) == 3:
        full_simulation = sys.argv[1]
        simulation_number = int(sys.argv[2])
    else:
        full_simulation = "True"
        simulation_number = sys.argv[1] if len(sys.argv) > 1 else 1
    
    time_limit = 500
    save_frame_interval = 10
    dt = 0.05
    velocity_profile_position_bin = 5
    simulation_name = f"simulation_{simulation_number}"
    globals_config_path = "configs/globals.py"
    simulation_config_path = f"configs/simulation_{simulation_number}.py"
    morphology_config_path = "configs/morphology_config.py"
    show_velocity_field = True

    if full_simulation == "True":
        simulation(time_limit, save_frame_interval, dt, globals_config_path, simulation_config_path, morphology_config_path, simulation_name, velocity_profile_position_bin, pertubation=False, show_velocity_field=show_velocity_field, rm_frames = True)
    else:
        simulation_only_for_window_size(time_limit, save_frame_interval, dt, globals_config_path, simulation_config_path, morphology_config_path, simulation_name, velocity_profile_position_bin, pertubation=False, show_velocity_field=show_velocity_field, rm_frames = True)


    