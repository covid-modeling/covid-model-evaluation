import numpy as np
import coveval.staging as staging


def test_exponential_has_not_left_exponential_stage():
    """
    Asserts that when called on an exponentially growing function, the stager does not declare exponential growth
    to be over.
    """
    expstager = staging.exponential_stager()
    y_true = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    is_different, p_value = expstager.test(y_true)
    assert not is_different
    assert p_value == 1

def test_leading_zeros_ignored():
    """
    Leading zeros are supposed to be ignored by the stager.
    """
    expstager = staging.exponential_stager()
    y_true = [12, 34, 1, 4, 64]
    y_true_padded = [0, 0, 0, 12, 34, 1, 4, 64]
    is_different, p_value = expstager.test(y_true)
    is_different_padded, p_value_padded = expstager.test(y_true_padded)
    assert is_different == is_different_padded
    assert p_value == p_value_padded

def test_linear_is_different_from_exponential():
    """
    Asserts that when called on a linearly growing function, the stager does not identify it as an exponential
    growth.
    """
    expstager = staging.exponential_stager()
    y_true = range(23) * np.asarray(2) + 3
    is_different, _ = expstager.test(y_true)
    assert is_different
