#!/usr/bin/env python3

#----------------------------------------------------------------------------------------------------------
# Warning: this script was hacked together to support a one-off experiment. Provided mostly for information
# purpose as it contains the grid and seed parameters used to run covid-sim calibration simulations.
#----------------------------------------------------------------------------------------------------------

import os
import re
import json

# This dictionary translates the efficiency level of social distancing into the corresponding residual contacts
cr_values = {
    'cr25': 0.75,
    'cr40': 0.60,
    'cr50': 0.50,
    'cr60': 0.40,
    'cr70': 0.30,
    'cr75': 0.25,
    'cr80': 0.20,
    'cr85': 0.15,
    'cr90': 0.10,
    'cr95': 0.05,
    }

# The below was correct as of beginning of June. As more interventions are added to each state, the dictinaries
# below will need updating.

# regex to capture the contact rates of interest
regexes = {
    'California': "(0.[\d]*)\t(0.[\d]*)",
    'Illinois': "([\d]*)\t(0.[\d]*)",
    'Massachusetts': "([\d]*)\t(0.[\d]*)",
    'Michigan': "([\d]*)\t(0.[\d]*)\t(0.[\d]*)",
    'New_Jersey': "([\d]*)\t(0.[\d]*)\t(0.[\d]*)",
    'New_York': "([\d]*)\t(0.[\d]*)\t(0.[\d]*)",
    }

# start with contact rate of 1 if first intervention period did not include social distancing
replace_prefixes = {
    'California': '',
    'Illinois':'1\\t',
    'Massachusetts':'1\\t',
    'Michigan':'1\\t',
    'New_Jersey':'1\\t',
    'New_York':'1\\t',
    }

# number of successive intervention periods that include social distancing
num_sds = {
    'California': 2,
    'Illinois': 1,
    'Massachusetts': 1,
    'Michigan': 2,
    'New_Jersey': 2,
    'New_York': 2,
    }

# The function below take an `input-params.txt` file and creates copies modified with the desired level of social
# distancing for each intervention period that include social distancing.
def modify_relative_contact_rates(filename, geo, output_dir="."):
    output_base = os.path.splitext(os.path.basename(filename))[0]
    
    # read in the file content
    with open(filename) as f:
        original = f.read()
        
    # build search string
    field = "Relative spatial contact rates over time given social distancing"
    regex = r"\[" + field + "\]\n" + regexes[geo]
        
    for cr, sd_value in cr_values.items():
        # build substitution string
        subst = "[" + field + "]\\n" + replace_prefixes[geo] + str(sd_value)
        for i in range(1, num_sds[geo]):
            subst += "\\t" + str(sd_value)
            
        # get new file content
        modified = re.sub(regex, subst, original, 0, re.MULTILINE)
        
        # save modified file
        out_filename = os.path.join(output_dir, geo, output_base + '_' + cr + '.txt')
        with open(out_filename, 'w') as f:
            f.write(modified)
            
# generate input-params for each state
base = './_param_files/'
for geo in ['California','Illinois','Massachusetts','Michigan','New_Jersey','New_York']:
    filename = os.path.join(base, geo, geo + '_input-params.txt')
    modify_relative_contact_rates(filename, geo, base)
    

# Write config files consumed by the covid-sim runner. Note that the paths referenced in the file made sense in the
# context of the covid-sim repo.
cfg = {
    "threads": 8,
    "pop_density_file": "populations/wpop_usacan.txt.gz",
    "network_seeds": [98798150, 729101],
    "num_runs": 10,
    "r0": [1.75, 2.0, 2.2, 2.4, 2.6, 2.8, 2.9, 3.0, 3.1, 3.3, 3.5, 3.75],
    "geographies":{},
    }

# create multiple config files instead of a giant one to make it easier to parallelise the jobs
out_dir = './_config_files/'
for geo in ['California','Illinois','Massachusetts','Michigan','New_Jersey','New_York']:
    for i, cr in enumerate(cr_values.keys()):
        # add references to relevant param files, note that the path below have to make sense in the context of
        # repo where the covid-simm runner will be called from
        tmp_cfg = dict(cfg)
        tmp_cfg['geographies'] = {
            geo: {
                "pre_param_file": os.path.join('param_files', geo, geo + '_pre-params.txt'),
                "param_files": [os.path.join('param_files', geo, geo + '_input-params_' + cr + '.txt')]
            }
        }
        
        # save config
        out_filename= os.path.join(out_dir, geo + '_' + str(i+1) + '.json')
        with open(out_filename, 'w') as f:
            json.dump(tmp_cfg, f, indent=4)
