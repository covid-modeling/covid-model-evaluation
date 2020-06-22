
import os
import json
import fnmatch
import pandas as pd

def load_predictions(filename, prefix='', cols=None, round_values=True):
    """Load predictions from the covasim model.
    
    Parameters
    ----------
    filename : str
        Path to .json file
    prefix : str, optional
        String to prepend all fields names.
    cols : [str], optional
        Names of columns to keep.
    round_values : boolean, optional
        If True (default), round the columns values to integers.
    
    Returns
    -------
    out : DataFrame
        A DataFrame of predictions with a DatetimeIndex.
    """
    with open(filename) as f:
        _raw = json.load(f)
    
    # prune irrelevant fields
    _raw_pruned = {k: v for k,v in _raw.items() if k not in ['t','timeseries_keys']}
    
    df = pd.DataFrame.from_dict(_raw_pruned)
    
    # set index
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # select subset of columns
    if cols is None:
        cols = df.columns
    df = df[cols]
    
    # round values
    if round_values:
        df = df.round().astype(int)

    # add prefix
    df.rename(columns={c: prefix + c for c in df.columns}, inplace=True)

    return df

def batch_load_predictions(base_path, filename='data.json', prefix='', cols=None, round_values=True, exclude=None):
    """Produce a nested dictionary reproducing the structure of the subdirectories in `base_path` and whose
    leaves are a DataFrame corresponding to a simulation results.
    """
    if exclude is None:
        exclude = []
    out = {}
    for d in os.listdir(base_path):
        d_full = os.path.join(base_path, d)
        if os.path.isdir(d_full) and d not in exclude:
            out[d] = batch_load_predictions(d_full, filename, prefix, cols, round_values, exclude)
        elif fnmatch.fnmatch(d, filename):
            out = load_predictions(d_full, prefix, cols, round_values)
    return out    
