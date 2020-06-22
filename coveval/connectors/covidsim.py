
import os
import fnmatch
import numpy as np
import pandas as pd

def load_predictions(filename, prefix='', cols=None, round_values=True):
    """Load raw predictions from the covid-sim model.
    
    Parameters
    ----------
    filename : str
        Path to a tsv file.
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
    df = pd.read_csv(filename, sep='\t')
    df.rename(columns={'t': 'date'}, inplace=True)
    df['date'] = np.asarray([pd.Timedelta(x, unit='day') for x in df['date']])
    df['date'] += pd.to_datetime('01-01-2020')
    
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

def batch_load_predictions(base_path, filename='*severity.xls', exclude=None, prefix='', cols=None):
    """Produce a nested dictionary reproducing the structure of the subdirectories in `base_path` and whose
    leaves are a DataFrame corresponding to a simulation results.
    """
    if exclude is None:
        exclude = []
    out = {}
    for d in os.listdir(base_path):
        d_full = os.path.join(base_path, d)
        if os.path.isdir(d_full) and d not in exclude:
            out[d] = batch_load_predictions(d_full, filename, exclude, prefix, cols)
        elif fnmatch.fnmatch(d, filename):
            out = load_predictions(d_full, prefix, cols)
    return out