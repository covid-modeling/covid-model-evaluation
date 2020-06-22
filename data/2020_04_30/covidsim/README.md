# Covidsim calibration simulations

## Disclaimer

The [covid-sim](https://github.com/mrc-ide/covid-sim) model takes many parameters, here we only describe a simple procedure to calibrate 2 such parameters:
- the `R0` value
- the effectiveness of social distancing measures 

These notes are succinct and assume some familiarity with the covid-sim model, refer to its [documentation](https://github.com/mrc-ide/covid-sim/blob/master/docs/inputs-and-outputs.md) if that's not the case.

## Procedure

For each state of interest we ran the following grid of simulations:
```latex
{R0} x {CR} x n
```

where:
- `R0` = R value at the beginning of the epidemic in the state
- `CR` = relative drop in contact rates due to social distancing measures in the state 
- `n` = number of repeats
 
Concretely, 1,200 simulations were performed for each state:
- `R0` in `[1.75, 2.0, 2.2, 2.4, 2.6, 2.8, 2.9, 3.0, 3.1, 3.3, 3.5, 3.75]`
- `CR` in `[0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]`
- `n = 10`

### Parameter files

The calibration date and the cumulative number of fatalities reported in the state at that time were used in the `pre-params.txt` of each state.

Publicly available information on the successive confinement and social distancing interventions was used to create a base `input-params.txt` for each state. The [generate_covidsim_params.py](generate_covidsim_params.py) script was then used to change the `[Relative spatial contact rates over time given social distancing]` field of that file so that the relative contact rates reflected the `CR` level of contact reductions. Note that this script is extremely rough, it's mostly provided for information purpose. These files are available in [_param_files/](_param_files/).

Do note that the **same** contact rate value was used for each intervention period that included social distancing. This means that covid-sim was not able to consider a scenario where social distancing remains in place but contact rates changed over time. This simplification avoided an explosion of the search space but of course can be improved on (notably iterative calibrating of the effectiveness of successive social distancing measures will become more relevant as time goes by).

### Running the simulations

The run `run_many.py` script available in a [public branch](https://github.com/mrc-ide/covid-sim/tree/matt-gretton-dann/full_run_sample.py/data) of the covid-sim repository was used to trigger the simulations.

The [generate_covidsim_params.py](generate_covidsim_params.py) script was used to generate the configuration files consumed by that script. These files are available in [_config_files/](_config_files/) but note that the relative paths and data they refer to are to be understood in the context of the covid-sim branch link above.
