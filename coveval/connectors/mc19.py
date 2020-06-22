
import os
import json
import fnmatch
import pandas as pd

# The code below is only valid for a single scenario - but it's easy enough to modify it to handle several.
def load_predictions(filename, prefix='', cols=None, round_values=True):
    """Load predictions in the format provided by the unified UI.
    
    Parameters
    ----------
    filename : str
        Path to .json file
    prefix : str, optional
        String to prepend all fields names.
    cols : [str], optional
        Names of the fields to keep.
    round_values : boolean, optional
        If True (default), round the columns values to integers.
    
    Returns
    -------
    out : DataFrame
        A DataFrame of predictions with a DatetimeIndex.
    """
    if cols is None:
        cols = ['dailyDeath']

    with open(filename) as f:
        _raw = json.load(f)
    _days = _raw['data']['location']['days']['data']    
    _data = {}
    for col in cols:
        for k, v in _raw['data']['location']['scenario'][col].items():
            _data[col + '-' + k] = v['data'][:len(_days)]
    _data['date'] = [pd.Timestamp('2020-01-01') + pd.Timedelta(d - 1, unit='day') for d in _days]
    df = pd.DataFrame.from_dict(_data)

    # set index
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # round values
    if round_values:
        df = df.round().astype(int)
        
    # add prefix
    df.rename(columns={c: prefix + c for c in df.columns}, inplace=True)

    return df

def batch_load_predictions(base_path, filename='data.json', exclude=None, prefix='', cols=None):
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
