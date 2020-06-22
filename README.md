![CI Tests](https://github.com/covid-modeling/covid-model-evaluation/workflows/CI%20Tests/badge.svg)

# Covid Model Evaluation

This repository shares code used to compare, calibrate and evaluate different prediction models of the Covid19 outbreak, facilitating reproducing or extending our results. This work is summarized in the report [How predictive are models of Covid-19?](how-predictive-are-models-of-covid-19.pdf).

:warning: This is work in progress released for the sake of transparency, and not a finished package.

## Getting started

We require Python 3 and a list of dependencies that can be installed with `pip install -r requirements.txt` from a clone of this repo.

Processing a model's prediction is explained in [a notebook](/notebooks/demo.ipynb).

## Elements

### Coveval

The Python module [Coveval](/coveval) encapsulates the main scientific functionality we used for our investigations.

Its purpose is to [smooth](#smoothing), [normalize](#normalizing), [stage](#staging) and [score](#scoring) time series such as fatality-per-day data.

#### Smoothing

Reported daily fatalities are very noisy. For many geographies, a substantial part of that noise seems to be reporting noise, e.g. due to cases being reported late (sometimes in a bunch). We provide several [smoothers](/coveval/core/smoothing.py) so as not to be thrown off, most relevantly:

* `coveval.core.smoothing.missed_cases`: Infers the actual day deaths occurred assuming some can be reported late and the real ground truth is likely to be smooth. Makes sense for unevenly reported ground truth, but likely needs more development work to be able to use this out of the box.
* `coveval.core.smoothing.gaussian`: A standard low-pass Gaussian filter: easier to use than missed cases. This is recommended for smoothing stochastic model _predictions_.
* `coveval.core.smoothing.identity`: No smoothing.

#### Normalizing

When comparing or evaluating models, we would like to be able to break down the mismatch between model and ground truth on a daily basis. 

In doing so, we may want to avoid "punishing a model twice" for getting a single point in the outbreak wrong. Rather, we might try to tease apart an initial imperfection from later conclusions that are incorrect on the face of it, but make sense relative to the initial mistake.

For example, assume that, in the beginning, the model is too pessimistic, predicting 100 infections at a time when there are in fact just 50. Assume furthermore that in the current state of the outbreak, infections double twice a week. If the model predicts this dynamic correctly, it will continue to overestimate the number of infections a week later (predicting 400 instead of 200). But that mistake is not "new", it's a follow on mistake, and the model arguably shows more promise than one that only predicts an increase of 100 to 200: even though the latter number happens to be correct, the dynamics are off.

Thus, we provide functionality to score not score the raw prediction, but the prediction that the model would likely have given knowing the previous actual extent of the outbreak. One simple heuristic for that would be the following: If the outbreak (due to imperfect modelling at the beginning) is only half the size that the model reported for the last week, then today it may make sense to scale its prediction downwards by ½. This is the basic idea behind the approach implemented in `coveval.core.normalising.dynamic_scaling`.

#### Staging

`coveval.staging` is a prototype to test whether an outbreak has passed its very initial exponential phase. If not, the user may well want to refrain from further calibration, since for most models, a large number of very similar parameter sets will be able to fit an exponential curve.

#### Scoring

A `coveval.scoring.scorer` applies both [smoothing](#smoothing) to the ground truth and [scaling](#scaling) to the prediction, and then computes a score. We provide several [losses](/coveval/core/losses.py), but have concentrated on `coveval.core.losses.poisson`, which is a pointwise loglikelihood, treating the model prediction as expectation value of a Poisson distribution. Some models do not report the expectation value but rather an insantiation of the random variable, in which case it may make sense to either smooth the prediction or average it with different runs before scoring it.

#### Connectors

`coveval.connectors` contains the functionality needed to parse and load the predictions made by the different models. The module names are self-explanatory, with the exception of `coveval.connectors.generic`, which corresponds to the output format specified in the [Covid Model-Runner JSON output schema](https://github.com/covid-modeling/model-runner/blob/master/packages/api/schema/output.json). This format is emitted by the [Covid Model-Runner](https://github.com/covid-modeling/model-runner) as the result of running one or more models. The user interface for invoking the Covid Model-Runner is available at https://covid-modeling.org/. Please [raise an issue in this project](https://github.com/covid-modeling/web-ui/issues) if you require access to the UI.


#### Other utilities

The `coveval.utils` module mostly contains data processing and plotting methods that we made frequent use of in our analysis.

It also contains a convenience method to retrieve ground truth data for the USA as reported on [covidtracking.com](https://covidtracking.com/).

### Data

The [data](/data/) folder contains simulations from the different models used in our analysis and documentation on how they were obtained.

### Notebooks

The [notebooks](/notebooks) folder contains material to reproduce our analysis and examples on how to use the `coveval` modules. 

## Questions, comments, and where to find us

- Found a bug? [Raise an issue!](https://github.com/covid-modeling/covid-model-evaluation/issues)
- Have a question? [Send an email!](mailto:covid-modeling+opensource@github.com)
- Want to contribute? [Raise a pull request!](https://github.com/covid-modeling/covid-model-evaluation/pulls)

## Contributing

We welcome contributions to this project from the community. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).
Some material in `data/` is used under other licenses, see `LICENSE` files in applicable subdirectories for details.