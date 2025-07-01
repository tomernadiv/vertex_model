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
                   dt=0.1,                # time step  
                   velocity_profile_position_bin=5,
                   do_cut = False):       # if we want to cut the tissue at t == 10

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

    #initialize a list of x velocities of the inner border layer
    vx_border_layer_series = []

    max_velocities = []

    # iterate overthe graph
    for t in range(0, time_limit):
        print(f"\n---------------------Time {t}---------------------")

        # --- Apply cut only once at t == 10 ---
        if t == 10 and do_cut == True:
            cut_x = T.num_cols // 4  # or any specific col you want
            row_start, row_end = 0, T.num_rows
            T.apply_vertical_cut(cut_x=cut_x, row_start=row_start, row_end=row_end)

        # compute velocities
        T.compute_all_velocities()

        # compute velocity profile
        position, velocity = T.calculate_velocity_profile(bin_length=velocity_profile_position_bin)
        velocity_profiles.append((position, velocity))
        max_velocities.append(max(abs(velocity)))
        # calculate mean velocity of inner bound layer
        vx_border = T.compute_inner_border_x_velocity_middle_band()
        vx_border_layer_series.append(vx_border)
        

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
           'area': np.array(area_percs),
           'velocity_profile_position': positions,
           'velocity_profile_velocity': velocities,
           'vx_inner_border' : vx_inner_border,
           'max_velocities': max_velocities
           }


    return res


def grid_search_on_final_area_experiment(simulation_number):
    """
    Runs a grid search experiment    """

    # create saving path
    saving_path = os.path.join(experiments_folder_path, f'grid_search_on_final_area_experiment_sim={simulation_number}')
    os.makedirs(saving_path, exist_ok=True)

    # define shit
    time_limit = 100
    dt = 0.5
    velocity_profile_position_bin = 5
    globals_config_path = "configs/globals.py"
    simulation_config_path = f"configs/simulation_{simulation_number}.py"
    morphology_config_path = "configs/morphology_config.py"

    # define grid search parameters
    if simulation_number == 1:
        d = {'expansion_const': np.linspace(1, 5, 5).tolist()}

    elif simulation_number == 2:
        d = {'push_out_force': np.linspace(1, 2, 2).tolist()}

    elif simulation_number == 3:
        d = {'shrinking_const': np.linspace(0.1, 1, 2).tolist()}

    # initiate Tissue
    T = tissue.Tissue(globals_config_path = globals_config_path, simulation_config_path = simulation_config_path, morphology_config_path = morphology_config_path)


    # set up results dict
    dependent_var = next(iter(d.keys()))
    results = {dependent_var: d[dependent_var], 'final_area': []}

    # run simulation
    for param_value in d[dependent_var]:
        T.force_constants_change({dependent_var: param_value})
        print(T.non_neuron_internal_rest_length)
        res = run_simulation_no_savings(T=T, time_limit=time_limit, dt=dt, velocity_profile_position_bin=velocity_profile_position_bin)
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
    plt.title(f'Grid Search Results for Simulation {simulation_number}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(saving_path, 'results_plot.png'))
    plt.show()



if __name__ == "__main__":
    grid_search_on_final_area_experiment(simulation_number=1)




