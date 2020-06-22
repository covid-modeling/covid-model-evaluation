# CovaSim Calibration and Test

## Disclaimer

These are notes on how to replicate a one-off proof-of-concept exploration. Both notes and code are rather rough. Please contact @wunderalbert with questions about either.

## Organisation and Sources

In this folder, we are using CovaSim's own calibration of their model. The procedure is [here](https://github.com/amath-idm/covasim_apps/blob/master/us_states/), with the output being [here](https://github.com/amath-idm/covasim_apps/tree/master/us_states/calibrated_parameters).

Such a calibration is not a deterministic process, so on recommendation we're running several of them using `run_calibrated_model.py` in order to aggregate the results.

## Prerequisites

Install CovaSim (`pip install covasim`) and `sciris`, both of which are on `pip`. 

Data, script and `load_data.py` should be in the same folder. The required epidemiological data can be obtained by cloning the CovaSim [Repo](https://github.com/InstituteforDiseaseModeling/covasim) and running their data scraper, i.e. in the CovaSim directory going `data/run_scrapers` -- it should appear in the folder `data/epi_data/covid-tracking/`. Copy paste the `csv`s generated there to the `data` subfolder of the folder containing `run_calibrated_model.py` and `load_data.py`.

## Running

Run `python run_calibrated_model.py`.

## License

The contents of this directory were produced using the [Covasim model](https://github.com/InstituteforDiseaseModeling/covasim)
and [covasim_apps](https://github.com/amath-idm/covasim_apps), which are licensed under the Creative Commons Attribution-ShareAlike 4.0 International Public License. See the [LICENSE](./LICENSE) file in this directory for details.

