import numpy as np
import coveval.core.losses as losses


def test_normal_scaled():
    """
    Asserts that the normalised loss is the same for different `(y_true, y_pred)` where the ratio
    `(y_true-y_pred)/y_pred` is constant.
    """
    # using default values
    ns = losses.normal_scaled()
    
    v1 = ns.compute_pointwise(25,50) - ns.compute_pointwise(50,50)
    v2 = ns.compute_pointwise(150,100) - ns.compute_pointwise(100,100)
    assert round(v1,7) == round(v2,7)
    assert round(v1,7) == round(-np.log(ns.rel_value),7)
    
    v1 = ns.compute_pointwise(55,50) - ns.compute_pointwise(50,50)
    v2 = ns.compute_pointwise(110,100) - ns.compute_pointwise(100,100)
    assert round(v1,7) == round(v2,7)
    
    # using custom values
    ns = losses.normal_scaled(delta_pc=0.1, rel_value=0.75)
    
    v1 = ns.compute_pointwise(45,50) - ns.compute_pointwise(50,50)
    v2 = ns.compute_pointwise(110,100) - ns.compute_pointwise(100,100)
    assert round(v1,7) == round(v2,7)
    assert round(v1,7) == round(-np.log(ns.rel_value),7)
    assert ns.rel_value == 0.75
    
    v1 = ns.compute_pointwise(100,50) - ns.compute_pointwise(50,50)
    v2 = ns.compute_pointwise(200,100) - ns.compute_pointwise(100,100)
    assert round(v1,7) == round(v2,7)
