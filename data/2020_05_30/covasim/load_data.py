import pandas  as pd
import sciris  as sc


#%% Initialize

mapping = {
    'CA':'California',
    'IL':'Illinois',
    'MA':'Massachusetts',
    'MI':'Michigan',
    'NJ':'New Jersey',
    'NY':'New York',
    'PA':'Pennsylvania',
}
states = list(mapping.keys())
folder = '~/covasim_apps/us_states/data'


def load_data():

    #%% Get the raw data
    data = sc.objdict()
    for state in states:
        data[state]  = sc.objdict()

    for state in states:
        filename = f'{folder}/{state}.csv'
        data[state].epi = pd.read_csv(filename)


    # From https://github.com/covid-modeling/covasim-connector/blob/master/runsim.py, originally http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv
    raw_pop_sizes = pd.read_csv(f'{folder}/nst-est2019-alldata.csv')
    for state in states:
        data[state].popsize = int(raw_pop_sizes[raw_pop_sizes.NAME == mapping[state]].POPESTIMATE2019)

    print('Done.')

    return data


if __name__ == '__main__':
    _ = load_data()