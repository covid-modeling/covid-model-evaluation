import pytest
import numpy as np
from coveval.core import normalising

@pytest.fixture
def y_true():
    return np.asarray([10, 2, 2, 4, 5, 56, 3, 2, 3, 2, 4, 54, 56, 72])

@pytest.fixture
def y_pred_long():
    return np.asarray([10, 2, 2, 4, 5, 500, 500, 500, 500, 500, 3, 2, 56, 56, 56, 3, 56, 4])

@pytest.fixture
def y_pred_short():
    return np.asarray([10, 2, 3, 2, 4, 56, 1])


def test_dynamic_offset_on_equality(y_true):
    """
    Asserts that when called with the same sample array twice, the `normalise` function will return it - i.e.
    that the `normalise(x, x)` is the identity function.
    """
    nml = normalising.dynamic_offset()

    # no averaging
    y_norm, _ = nml.normalise(y_true, y_true, size=1, verbose=True)
    assert np.all(np.equal(y_true, y_norm))

    # averaging over last 2 observations
    y_norm, _ = nml.normalise(y_true, y_true, size=2, verbose=True)
    assert np.all(np.equal(y_true, y_norm))

    # averaging over last 10 observations
    y_norm, _ = nml.normalise(y_true, y_true, size=10, verbose=True)
    assert np.all(np.equal(y_true, y_norm))

    # averaging over last 10 observations, shorter window
    y_norm, _ = nml.normalise(y_true, y_true, size=10, window=5, verbose=True)
    assert np.all(np.equal(y_true, y_norm))
    
def test_dynamic_offset_input_lengths(y_true, y_pred_short, y_pred_long):
    """
    Checks that the normalise function runs out of values when expected depending on lentgh inputs and parameters.
    """
    nml = normalising.dynamic_offset()
    
    # too few predictions
    with pytest.raises(ValueError):
        nml.normalise(y_true, y_pred_short, size=1, window=3, verbose=True)
    with pytest.raises(ValueError):
        nml.normalise(y_true, y_pred_long, size=1, window=7, verbose=True)
    
    # enough predictions
    y_norm, _ = nml.normalise(y_true, y_pred_short, size=1, window=2, verbose=True)
    y_norm_expected = np.asarray([10., 2., 3., 3., 2., 2., 2., 2., 4., 56., 56., 56., 1., 1.])
    assert np.all(np.equal(y_norm, y_norm_expected))
    
    y_norm, _ = nml.normalise(y_true, y_pred_long, size=1, window=4, verbose=True)
    y_norm_expected = np.asarray([10., 2., 2., 4., 5., 500., 500., 500., 500., 500., 500., 500., 500., 500.])
    assert np.all(np.equal(y_norm, y_norm_expected))

def test_dynamic_scaling_on_equality(y_true):
    """
    Asserts that when called with the same sample array twice, the `normalise` function will return it - i.e.
    that the `normalise(x, x)` is the identity function.
    """
    nml = normalising.dynamic_scaling()

    y_norm, _ = nml.normalise(y_true, y_true)
    assert np.all(np.equal(y_true, y_norm))
