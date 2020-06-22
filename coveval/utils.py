import scipy.signal
import scipy.ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

# data ingestion tools
#---------------------
def get_outbreak_data_usa(filename=None, fillna=None, prefix=''):
    """
    Load available data about the oubtreak for US states in the format provided by https://covidtracking.com/.
    
    Parameters
    ----------
    filename : str, optional
        Path to csv file or url. If unspecified, defaults to 'https://covidtracking.com/api/v1/states/daily.csv'
    fillna : optional
        If specified, fill missing values with this value.
    prefix : str, optional
        String to prepend all fields names.
        
    Returns
    -------
    out : DataFrame
        A DataFrame with a MultiIndex (geography, date).
    """
    if filename is None:
        filename = 'https://covidtracking.com/api/v1/states/daily.csv'
    out = pd.read_csv(filename)
    if fillna is not None:
        out.fillna(0, inplace=True)
    
    # we build a MultiIndex with levels ['geography','date'] where 'date' is a DatetimeTindex.
    out.rename(columns={'state': 'geography'}, inplace=True)
    out['geography'] = out['geography'].apply(lambda x: 'US-' + x) # use ISO 3166 code for state name
    out['date'] = pd.to_datetime(out['date'], format="%Y%m%d")
    
    # set index
    out.set_index(['geography','date'], inplace=True)
    out.sort_index(ascending=[True, True], inplace=True)
    
    # add prefix
    out.rename(columns={c: prefix + c for c in out.columns}, inplace=True)
    return out
 
def add_outbreak_data(df, outbreak, fillna=None):
    """
    Convenience method to merge reported outbreak data to predictions.
    
    Parameters
    ----------
    df : DataFrame
        A DataFrame of predictions with a DatetimeIndex.
    outbreak : DataFrame, optional
        A DataFrame of reported data with a DatetimeIndex.
    fillna : boolean, optional
        If specified, fill missing values with this value.

    Returns
    -------
    out : DataFrame
        A DataFrame with a DatetimeIndex.
    """
    out = df.merge(outbreak, left_index=True, right_index=True, how='left')
    if fillna is not None:
        out.fillna(fillna, inplace=True)
    return out

# summary statistics
#-------------------
def calc_quantiles_over_dfs(dfs, col, q=None):
    """
    Compute quantiles for a given column over several DataFrames.

    Parameters
    ----------
    dfs : [DataFrame] or {k: DataFrame}
        A sequence of DataFrame, all containing a column `col`. If a dict is specified its values will be taken as
        the sequence (its keys are ignored).
    col : str
        The name of the column for which to compute the quantiles.
    q : [float], optional
        The quantiles to compute. Defaults to [0.1, 0.5, 0.9].
        
    Returns
    -------
    A DataFrame with the same index as the input DataFrames and columns `${col}-{$qi}` for each value `qi` in `q`.
    """
    if isinstance(dfs, dict):
        dfs = dfs.values()
    if q is None:
        q = [0.1, 0.5, 0.9]
    out = pd.concat([df[col] for df in dfs], axis=1, join='inner').quantile(q=q, axis=1).transpose()
    out.rename(columns={q_val: col + '-' + str(q_val) for q_val in q}, inplace=True)
    return out

def calc_mean_over_dfs(dfs, col):
    """
    Compute mean of a given column over several DataFrames.

    Parameters
    ----------
    dfs : [DataFrame] or {k: DataFrame}
        A sequence of DataFrame, all containing a column `col`. If a dict is specified its values will be taken as
        the sequence (its keys are ignored).
    col : str
        The name of the column for which to compute the mean.
        
    Returns
    -------
    A DataFrame with the same index as the input DataFrames and a `${col}-mean` column.
    """
    if isinstance(dfs, dict):
        dfs = dfs.values()
    out = pd.concat([df[col] for df in dfs], axis=1, join='inner')
    out[col + '-mean'] = out.mean(axis=1)
    return out[[col + '-mean']]

# visualisation tools
#--------------------
def show_data(df, cols, scatter=None, fill_between=None, t_min=None, t_max=None, show_times=None, y_log=False,
              ax=None, x_label=None, y_label=None, y_max=None, title=None, title_font=None, date_auto=True,
              show_leg=True, colors=None, linestyles=None, linewidths=None, figsize=(15,5)):
    """
    Utility function to plot data contained in a DataFrame, usually with a DatetimeIndex. Keywords and
    functionality grow over time as needs evolve.
    
    Parameters
    ----------
    df : DataFrame
        A DataFrame whose index represents time.
    cols : [str]
        Name of columns to plot.
    scatter : [str], optional
        Use a scatter plot for these columns.
    fill_between : [2-tuple], optional
        Shade areas corresponding to values falling in between the specified columns values.
    t_min, t_max : index
        Minimum and maximum values of the DataFrame index to use (both inclusive).
    show_times : [], optional
        Show a vertical line for each of these index values. Strings are fine for DatetimeIndex.
    y_log : boolean, optional
        If True use a symlog scale for y axis (i.e. linear up to y=10 and log afterwards).
    ax : matplotlib.axes, optional
        Data will be plotted on `ax` if specified. Otherwise a new figure and `ax` object is created.
    x_label, y_label : str, optional
        Labels of axes.
    y_max : float, optional
        Boundary on the y scale.
    title : str, optional
        Title of axes.
    title_font : dict, optional
        Properties of title, e.g. {'fontsize': 20, 'fontweight': 'bold'}
    date_auto : boolean, optional
        If True use AutoDateLocator for positioning ticks on x-axis.
    show_leg : boolean or dict, optional
        If True or dict, show legend. If dict, can be used to specify the legend labels.
    colors : dict, optional
        Nested dictionary that can be used to specify the colours of `fill_between` or elements in `cols` and
        `scatter`, e.g. : ```colors={'fill_between': ['r', 'g], 'cols': {'pred': 'k'}, 'scatter': {'pred': 'k}}```.
        Note that, if specified, the length of the 'fill_between' list of colours must match that of the
        `fill_between` parameter.
    linestyles : dict, optional
        Line style to use for the specified columns.
    linewidths : dict, optional
        Line width to use for the specified columns.
    figsize : (float,float), optional
        Dimensions of figure to create if `ax` is not specified.

    Returns
    -------
    A dictionary {str: matplotlib artist} that can be used to create legend objects on a different graph.
    """

    # data slice of interest
    df = df.sort_index(ascending=True)
    if t_min is None:
        t_min = df.index.values[0]
    if t_max is None:
        t_max = df.index.values[-1]
    df = df.loc[t_min:t_max]

    # handle scatter and show_times
    cols = list(cols) #to avoid modifying the parameter
    if scatter is None:
        scatter = []
    for sc in scatter:
        if sc not in cols:
            cols.insert(0, sc)
    if show_times is None:
        show_times = []
            
    # handle default line style and colors
    if colors is None:
        colors = {}
    for k in ['cols','scatter']:
        if k not in colors:
            colors[k] = {}
    if linewidths is None:
        linewidths = {}
    if linestyles is None:
        linestyles = {}
    for l in cols + show_times:
        if l not in linewidths:
            linewidths[l] = 3
        if l not in linestyles:
            linestyles[l] = '-' if l in cols else '--'
            
    # handle legend specifications
    labels = {k: k for k in df.columns}
    if isinstance(show_leg, dict):
        for k, v in show_leg.items():
            labels[k] = v
        show_leg = True
        
    # create new figure if axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    
    # plot lines and scatter data
    lines = {}
    for i, col in enumerate(cols):
        if col in scatter:
            c = colors['scatter'][col] if col in colors['scatter'] else 'C'+str(i)
            lines[col] = ax.scatter(df.index, df[col], label=labels[col], color=c, alpha=0.4, marker='o')
        else:
            c = colors['cols'][col] if col in colors['cols'] else 'C'+str(i)
            lw = linewidths[col]
            ls = linestyles[col]
            alpha = 0.5
            lines[col] = ax.plot(df.index, df[col], label=labels[col], color=c, alpha=alpha, lw=lw, ls=ls)[0]
    
    # plot shaded area if requested
    if fill_between is not None:
        for j, fb in enumerate(fill_between):
            c = colors['fill_between'][j] if 'fill_between' in colors else 'C'+str(i+j)
            ax.fill_between(df.index, df[fb[0]], df[fb[1]], alpha=0.2, color=c)

    # plot vertical lines at specified times
    for t in show_times:
        ax.axvline(pd.to_datetime(t), linestyle=linestyles[t], linewidth=linewidths[t], color='k', alpha=0.4)
    
    # change to log scale if requested
    if y_log:
        ax.set_yscale('symlog', linthreshy=10)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    # deal with legend
    if show_leg:
        ax.legend(loc='upper right')
    
    # add labels and title
    ax.set_title(title, fontdict=title_font)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)    
    
    # set boundary on y axis if specified
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    # configure time axis
    if isinstance(df.index, pd.DatetimeIndex):
        if date_auto:
            locator = mdates.AutoDateLocator()
            formatter = mdates.AutoDateFormatter(locator)
            ax.set_xlim(mdates.date2num([pd.to_datetime(t_min), pd.to_datetime(t_max)]))
        else:
            ax.xaxis.set_minor_locator(mdates.DayLocator(range(1,32)))
            locator = mdates.WeekdayLocator(mdates.MO)
            formatter = mdates.DateFormatter("%d %b")
            ax.set_xlim(mdates.date2num([pd.to_datetime(t_min), pd.to_datetime(t_max)]))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    return lines

def show_normalisation(df, truth, pred_raw, pred_norm, pred_match, truth_smoothed=None, y_log=False,
                       y_lbl='# daily fatalities', x_pad=3, cb_center=0, figsize=(15,8)):
    '''
    Utility function to visualise the effect of the normalisation procedure on a model predictions, where the
    `pred_match` values are to be suitably computed depending on the normalising procedure applied.
    '''
    # find when truth was available
    idx = df.index[~df[truth].isnull()]
    
    # plot time series
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(idx, df.loc[idx, truth], marker='x', color='C0', label='reported')
    ax.plot(idx, df.loc[idx, pred_raw], color='k', alpha=0.3, label='predicted (raw)')
    ax.plot(idx, df.loc[idx, pred_norm], color='k', alpha=0.8, linestyle='--', label='predicted (normalised)')
    if truth_smoothed is not None:
        ax.plot(idx, df.loc[idx, truth_smoothed],color='C0', alpha=0.4, linestyle='--',
                label='reported (smoothed)')
    if y_log:
        ax.set_yscale('symlog', linthreshy=10)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    
    # add background colouring    
    _abs_max = max(np.abs(df.loc[idx, pred_match].values))
    divnorm = mcolors.TwoSlopeNorm(vmin=-_abs_max, vcenter=cb_center, vmax=_abs_max)
    pcf = ax.pcolorfast(mdates.date2num([idx[0], idx[-1]]), ax.get_ylim(),
                        df.loc[idx, pred_match].values[np.newaxis],
                        cmap='bwr', alpha=0.2, norm=divnorm)
    t_delta = pd.Timedelta(x_pad, unit='days')
    ax.set_xlim(idx[0]-t_delta, idx[-1]+t_delta)
    ax.set_ylabel(y_lbl)
    
    # configure colorbar
    cbar = fig.colorbar(pcf)
    cbar.ax.tick_params(size=0)
    cbar.set_ticks([-_abs_max, cb_center, _abs_max])
    cbar.set_ticklabels(['too pessimistic\n(reality better)','matches reality','too optimistic\n(reality worse)'])

    # configure time axis
    ax.xaxis.set_minor_locator(mdates.DayLocator(range(1,32)))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    
    return fig

def batch_show_data(geo, params, data, scores, cols, scatter=None, fill_between=None, colors=None,
                              t_min='2020-02-27', t_max=None, prop=None, highlight='min', highlight_num=1,
                              figsize=None, headers=None, **kwargs):
    """
    Convenience method to plot an array of simulations predictions and easily compare them. 
    
    Parameters
    ----------
    geo : str
        Key of geography for which to plot predictions.
    params : array-like
        2D-arrays of the parameters to plot, organised by rows, e.g. `[['2.0','2.1','2.2'],['3.0','3.1','3.2']]`
        to plot an array with 2 rows and 3 columns.
    data : dict
        Nested dictionary where `data[geo][param]` is a DataFrame with a DatetimeIndex for each element of
        `params`.
    scores : dict
        Nested dictionary where `scores[geo][param]` is a float for each element of `params`.
    cols : [str]
        Columns of the DataFrames to plot as lines (see show_data()).
    scatter : [str], optional
        Columns of the DataFrames to plot as scatter (see show_data()).
    fill_between : [2-tuple], optional
        Shade areas corresponding to values falling in between the specified columns values (see show_data()).
    colors : dict, optional
        See show_data().    
    t_min, t_max : str, optional
        Specifies data interval in the DataFrame to take into account and the x-axis limits (both inclusive). If
        t_max is set to None it defaults to the current date plus one week.
    prop : dict, optional
        Properties dictionary to define axes facecolor and title font and weight. Defaults to:
        ```{'title_font': {'fontsize': 20}, 'facecolor': 'w'}```
    highlight : dict or str, optional
        Use this to modify the properties of specified plots. If a dictionary must be of the form `{param: prop}`
        where `prop` is a property dictionary.
        If 'min' or 'max' the following properties will be applied to highlight the plots corresponding to the
        parameters with the lowest/highest score:
        ```{'title_font': {'fontweight': 'bold', 'fontsize': 20}, 'facecolor': '#f0faed'}```
    highlight_num : int, optional
        Use in conjunction with `highlight=min` or `highlight=max` to highlight the n lowest/highest score plots.
    figsize : 2-tuple or float, optional
        If a tuple specifies directly the figure size. If a float scales the default figure size. If unspecified
        the figure size is determined automatically based on the number of rows and columns.
    headers : dict, optional
        A dictionary of the form ```['rows': [str], 'cols': [str]]``` where the sequence of strings correspond
        to the headers of rows/columns. Both 'rows' and 'cols' or just one can be specified, either way it is
        necessary to make sure the length of the list(s) specified matches the number of rows/columns.
    """
    params = np.asarray(params)
    nrows = params.shape[0]
    ncols = params.shape[1]
    if figsize is None:
        figsize = 1
    if type(figsize) in [int, float]:
        figsize = (8*ncols*figsize, 5.5*nrows*figsize)    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    default_prop = {'title_font': {'fontsize': 20}, 'facecolor': 'w'}
    if prop is None:
        prop = {}
    for k, v in prop.items():
        default_prop[k] = v
    if highlight is None:
        highlight = {}
    elif highlight in ['min','max']:
        sort_reverse = False if highlight == 'min' else True        
        params_highlight = sorted(scores[geo].items(), key=lambda item: item[1],
                                  reverse=sort_reverse)[:highlight_num]
        highlight = {p[0]: {'title_font': {'fontweight': 'bold', 'fontsize': 20}, 'facecolor': '#f0faed'}
                     for p in params_highlight}        
    if t_max is None:
        t_max = pd.to_datetime('today') + pd.Timedelta(7, unit='day')
        t_max = t_max.strftime('%Y-%m-%d')
    
    for r in range(nrows):
        for c in range(ncols):
            # get corresponding style
            param = params[r,c]
            if param not in data[geo].keys():
                continue
            param_prop = dict(default_prop)
            if param in highlight:
                for k, v in highlight[param].items():
                    param_prop[k] = v
            axes[r,c].set_xlim(mdates.date2num([pd.to_datetime(t_min), pd.to_datetime(t_max)]))
            axes[r,c].set_facecolor(param_prop['facecolor'])
                        
            # plot data
            _score = scores[geo][param]
            if not isinstance(_score, str):
                _score = str(round(_score, 3))
            show_data(df=data[geo][param].loc[t_min:t_max], cols=cols, scatter=scatter, fill_between=fill_between,
                      colors=colors, ax=axes[r,c], show_leg=False, date_auto=True,
                      title_font = param_prop['title_font'],
                      title=param + ' (' + _score + ')', **kwargs)
            
    # deal with rows/columns headers
    if headers is not None:
        pad = 5
        if 'rows' in headers:
            h_rows = headers['rows']
            if len(h_rows) != nrows:
                print('Ignoring rows headers: wrong number of elements.')
            else:
                for ax, h_row in zip(axes[:,0], h_rows):
                    ax.annotate(h_row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                                textcoords='offset points', size=28, ha='right', va='center')
        if 'cols' in headers:
            h_cols = headers['cols']
            if len(h_cols) != ncols:
                print('Ignoring columns headers: wrong number of elements.')
            else:
                for ax, h_col in zip(axes[0], h_cols):
                    ax.annotate(h_col, xy=(0.5, 1.1), xytext=(0, pad), xycoords='axes fraction',
                                textcoords='offset points', size=28, ha='center', va='baseline')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.3)
    fig.suptitle(geo, size=26, weight='bold')
    return

# post-processing tools
#----------------------
def find_epidemic_peaks(values, filter, **kwargs):
    """
    A simple utility to identify peaks in epidemiological curves that works by applying a low-pass Gaussian filter
    on the predictions.

    Parameters
    ----------
    values : array
        The noisy values in which to identify peaks.
    filter : float
        Standard deviation of the Gaussian kernel applied for low-pass filtering.
    **kwargs : kwargs
        Arguments to be passed to `scipy.signal.find_peaks()`.
    
    Returns
    -------
    values_filtered : array
        Predictions with high-frequency noise filtered out.
    peaks : 2-tuple
        Outcome of `scipy.signal.find_peaks`.
    """
    values_filtered = scipy.ndimage.gaussian_filter1d(values, sigma=filter)
    return values_filtered, scipy.signal.find_peaks(values_filtered, **kwargs)
