import sciris as sc
import covasim as cv
import load_data as ld
import os as os

cv.check_version('1.5.0', die=True)

# we use vb to keep functions identical to source
vb = sc.objdict()
vb.verbose = 0

# The simulation is stochastic; how often should it be realized?
n_runs_per_parameter_set = 50

def create_sim_from_calibrated_pars(filename):
    '''Wrapper around create_sim that reads the parameters from a file'''
    pars_calib = sc.loadjson(filename);
    # Take care to use the same order that create_sim expects!
    pars = [pars_calib["pop_infected"],
            pars_calib["beta"],
            pars_calib["beta_day"],
            pars_calib["beta_change"],
            pars_calib["symp_test"]]
    return create_sim(pars)

def create_sim(x, vb=vb):
    ''' Create the simulation from the parameters '''

    # Convert parameters
    pop_infected = x[0]
    beta         = x[1]
    beta_day     = x[2]
    beta_change  = x[3]
    symp_test    = x[4]

    # Create parameters
    pop_size = 200e3
    pars = dict(
        pop_size     = pop_size,
        pop_scale    = data.popsize/pop_size,
        pop_infected = pop_infected,
        beta         = beta,
        start_day    = '2020-03-01',
        end_day      = '2021-05-30', # Run for at least a year
        rescale      = True,
        verbose      = vb.verbose,
    )

    #Create the sim
    sim = cv.Sim(pars, datafile=data.epi)

    # Add interventions
    interventions = [
        cv.change_beta(days=beta_day, changes=beta_change),
        cv.test_num(daily_tests=sim.data['new_tests'].dropna(), symp_test=symp_test),
        ]

    # Update
    sim.update_pars(interventions=interventions)

    return sim

def get_verbose_state_name(name):
    state_names = {
        'NY': 'New_York',
        'NJ': 'New_Jersey',
        'MI': 'Michigan',
        'CA': 'California',
        'MA': 'Massachusetts',
        'IL': 'Illinois'
    }
    return state_names[name]

if __name__ == '__main__':
    all_data = ld.load_data()
    states = ['NY', 'NJ', 'MI', 'CA', 'MA', 'IL']
    for state in states
        state_verbose = get_verbose_state_name(state)
        calibration_parameter_file = state_verbose + "/500/calibration_0/calibrated_parameters.json"

        print(f'Running {n_runs_per_parameter_set} simulations for {state_verbose} ({state}).')

        data = all_data[state]

        os.mkdir(state_verbose)
        for i_run in range(n_runs_per_parameter_set):
            sim = create_sim_from_calibrated_pars(calibration_parameter_file)

            sim.set_seed(sim['rand_seed'] + i_run)
            sim.run(until=600) # run for one year into the future

            file_folder = f'{state_verbose}/500/calibration_0/run_{i_run}/'
            os.mkdir(file_folder)

            sim.export_results(filename=f'{file_folder}results.json')
            print(".", end='', flush=True)
        print("")

print('All done.')
