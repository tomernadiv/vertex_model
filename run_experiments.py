from configs.run_config import *
import tissue
from configs.imports import *
from neuron_initiation import get_frame_bounds_from_index
from run_simulation import *
import pandas as pd



experiments_folder_path = os.path.join(os.getcwd(), 'experiments')


def run_simulation_no_savings(
                   T:tissue.Tissue,       # tissue object
                   time_limit:int,        # number of timestamps
                   dt=0.1):                # time step

    # init empty stacks
    # energies = []
    area_percs = []
    # velocity_profiles = []

    #init energy
    initial_energy = T.compute_total_energy()
    # energies.append(initial_energy)

    #init area
    initial_total_area = T.compute_total_area(window_only=False)
    initial_window_area = T.compute_total_area(window_only=True)
    area_percs.append(100)

    #initialize a list of x velocities of the inner border layer
    # vx_border_layer_series = []

    # max_velocities = []

    # iterate overthe graph
    for t in range(0, time_limit):
        # print(f"\n---------------------Time {t}---------------------")

        # --- Apply cut only once at t == 10 ---
        # if t == 10 and do_cut == True:
        #     cut_x = T.num_cols // 4  # or any specific col you want
        #     row_start, row_end = 0, T.num_rows
        #     T.apply_vertical_cut(cut_x=cut_x, row_start=row_start, row_end=row_end)

        # compute velocities
        # T.compute_all_velocities()

        # compute velocity profile
        # position, velocity = T.calculate_velocity_profile(bin_length=velocity_profile_position_bin)
        # velocity_profiles.append((position, velocity))
        # max_velocities.append(max(abs(velocity)))
        # # calculate mean velocity of inner bound layer
        # vx_border = T.compute_inner_border_x_velocity_middle_band()
        # vx_border_layer_series.append(vx_border)
        

        # Update for next iteration
        T.compute_all_forces()
        T.update_positions(dt=dt)
        T.update_heights()

        # add to stacjs
        # energies.append(T.compute_total_energy())
        area_percs.append((T.compute_total_area()/initial_window_area) * 100)

    # # stack up results
    # positions = [np.array(vp[0]) for vp in velocity_profiles]
    # velocities = [np.array(vp[1]) for vp in velocity_profiles]
    # vx_inner_border = np.array(vx_border_layer_series)
    res = {
           'area': np.array(area_percs),
        #    'energy': np.array(energies),
        #    'velocity_profile_position': positions,
        #    'velocity_profile_velocity': velocities,
        #    'vx_inner_border' : vx_inner_border,
        #    'max_velocities': max_velocities
           }


    return res


def grid_search_on_final_area_experiment(simulation_number):
    """
    Runs a grid search experiment    """

    # create saving path
    saving_path = os.path.join(experiments_folder_path, f'grid_search_on_final_area_experiment_sim={simulation_number}')
    os.makedirs(saving_path, exist_ok=True)

    # define shit
    time_limit = 300
    dt = 0.1
    globals_config_path = "configs/globals.py"
    simulation_config_path = f"configs/simulation_{simulation_number}.py"
    morphology_config_path = "configs/morphology_config.py"

    # define grid search parameters
    if simulation_number == 1:
        d = {'expansion_const': np.linspace(1, 3, 20).tolist()}

    elif simulation_number == 2:
        d = {'push_out_force_strength': np.linspace(1, 7, 20).tolist()}

    elif simulation_number == 3:
        d = {'shrinking_const': [0.3]}

    # set up results dict
    dependent_var = next(iter(d.keys()))
    results = {dependent_var: d[dependent_var], 'final_area': []}

    # run simulation
    for param_value in d[dependent_var]:
        # initiate Tissue
        T = tissue.Tissue(globals_config_path = globals_config_path, simulation_config_path = simulation_config_path, morphology_config_path = morphology_config_path)
        print(f"Running simulation with {dependent_var} = {param_value}")
        T.force_constants_change({dependent_var: param_value})
        print(T.non_neuron_internal_rest_length)
        res = run_simulation_no_savings(T=T, time_limit=time_limit, dt=dt)
        final_area = res['area'][-1]
        results['final_area'].append(final_area)

        # print results
        print(f"Parameter {dependent_var} = {param_value}, Final Area = {final_area}")

    # save dict as dataframe csv
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(saving_path, 'results.csv'), index=False)

    # fit a 2nd order polynomial to the results, plot and save the plot
    x = df[dependent_var].values
    y = df['final_area'].values
    coeffs = np.polyfit(x, y, 2)
    poly_fit = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_fit(x_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Data Points')
    formula = f"y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
    plt.plot(x_fit, y_fit, '-', label=formula)
    plt.axhline(y=300, color='r', linestyle='--', label='Final Area = 300%')
    plt.xlabel(dependent_var)
    plt.ylabel('Final Area (%)')
    plt.title(f'{dependent_var} vs Final Window Area')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(saving_path, 'results_plot.png'))
    plt.show()


def num_layers_vs_final_area(simulation_number):
    """
    Runs a grid search experiment    """

    # create saving path
    saving_path = os.path.join(experiments_folder_path, f'num_layers_vs_area_sim={simulation_number}')
    os.makedirs(saving_path, exist_ok=True)

    # define shit
    time_limit = 500
    dt = 0.1
    globals_config_path = "configs/globals.py"
    simulation_config_path = f"configs/simulation_{simulation_number}.py"
    morphology_config_path = "configs/single_window_morphology_config.py"

    # set up results dict
    num_layers_values = [x for x in range(26, 9, -1)]
    window_side = [60-2*x for x in num_layers_values]
    results = {
        'window_side': window_side,
        'final_area': []
    }

    # run simulation
    for num_layers in num_layers_values:

        # overwrite num layers
        with open(morphology_config_path, 'a') as f:
            f.write(f"\nnum_layers = {num_layers}  # temporary override\n")

        # initiate Tissue
        T = tissue.Tissue(globals_config_path = globals_config_path, simulation_config_path = simulation_config_path, morphology_config_path = morphology_config_path)
        T.plot_tissue(color_by='area')
        print(f"Running simulation with num_layers = {T.num_layers}")

        # run simulation
        res = run_simulation_no_savings(T=T, time_limit=time_limit, dt=dt)
        T.plot_tissue(color_by='area')
        final_area = res['area'][-1]
        results['final_area'].append(final_area)
        print(f"Final Area = {final_area}")

        # remove the temporary override
        with open(morphology_config_path, 'r') as f:
            lines = f.readlines()
        with open(morphology_config_path, 'w') as f:
            f.writelines(lines[:-1])     # Remove last line

    # save dict as dataframe csv
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(saving_path, 'results.csv'), index=False)

    # fit a 2nd order polynomial to the results, plot and save the plot
    x = df['window_side'].values
    y = df['final_area'].values
    coeffs = np.polyfit(x, y, 2)
    poly_fit = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_fit(x_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Data Points')
    formula = f"y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
    plt.plot(x_fit, y_fit, '-', label=formula)
    plt.xlabel('Window Side Length')
    plt.ylabel('Final Area (%)')
    plt.title(f'Window Side Length vs Final Window Area [simulation {simulation_number}]')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(saving_path, 'results_plot.png'))
    plt.show()

if __name__ == "__main__":
    simulation_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    grid_search_on_final_area_experiment(simulation_number=simulation_number)




