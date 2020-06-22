import numpy as np
from .core import smoothing
from .core import normalising
from .core import losses

class scorer:
    """
    Parameters
    ----------
    smoother: coveval.smoothing.smoother instance
        Used to smooth the truth. Defaults to identity smoother.
    normaliser : coveval.normalising.normaliser instance
        Used to normalise the time series. Defaults to identity normaliser.
    loss : coveval.losses.loss instance
        Used to compute the fit between truth and prediction.
    """
    def __init__(self, smoother=None, normaliser=None, loss=None):
        if smoother is None:
            smoother = smoothing.identity()
        if normaliser is None:
            normaliser = normalising.identity()
        if loss is None:
            loss = losses.poisson()
        self.smoother = smoother
        self.normaliser = normaliser
        self.loss = loss

    def score(self, y_true, y_pred, **kwargs):
        """
        Computes an overall score for a predicted time series by comparing it to the ground truth.
        
        Parameters
        ----------
        y_true : array
            Time series of observed data.
        y_pred : array
            Time series of predictions. Must be of same length as `y_true`.

        Returns
        -------
        A dictionary containing the `y_true` and `y_pred` entries as well as:
        `y_true_smoothed` : array
            True values after smoothing
        `y_pred_norm` : array
            Normalised predictions
        `y_pred_reg` : array
           Normalisation info for each time point.
        `y_pred_loss` : array
            Pointwise loss at each time point.
        `score` : scalar
            Overall loss.
        """

        # smooth inputs
        y_true_smoothed = self.smoother.smooth(y_true)

        # normalise predictions
        y_pred_norm, y_pred_reg = self.normaliser.normalise(y_true_smoothed, y_pred, **kwargs)

        # compute corresponding loss
        y_pred_loss = self.loss.compute(y_true_smoothed, y_pred_norm)

        # build results dictionary
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_true_smoothed': y_true_smoothed,
            'y_pred_norm': y_pred_norm,
            'y_pred_reg': y_pred_reg,
            'y_pred_loss': y_pred_loss,
            'score': np.mean(y_pred_loss)
            }
        return results
    
    def score_df(self, df, col_truth, col_pred, t_min=None, t_max=None, inplace=False, **kwargs):
        """
        Similar to `score()` method but accepts a Dataframe with a DatetimeIndex and handles NaNs. 
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame with a DatetimeIndex.
        col_truth : str
            The column containing the reported cases data.
        col_pred : str
            The column containing the corresponding predictions.
        t_min : str, optional
            If specified, only compute score from this index value (inclusive).
        t_max : str, optional
            If specified, only compute score  up to this index value (inclusive).
        inplace : bool, optional
            If True, modifies input `df`.
            
        Returns
        -------        
        A dictionary containing the following entries, computed taken the specified time bounds into account:
        'idx': DatetimeIndex
            Index values for which `col_truth` was available.
        `y_true` : array
            Reported values.
        `y_pred` : array
            Corresponding predictions.
        `y_true_smooth` : array
            Reported values after smoothing.
        `y_pred_norm` : array
            Normalised predictions.
        `y_pred_reg` : array
            Normalisation info at each timepoint.
        `y_pred_loss` : array
            Pointwise loss at each time point.
        `score` : scalar
            Overall loss.
            
        Also update the input DataFrame if inplace=True.
        """
        # don't modify input dataframe unless specified
        if not inplace:
            df = df.copy(deep=True)

        # smooth, normalise and compute loss
        self.smoother.smooth_df(df, col_truth, inplace=True)
        self.normaliser.normalise_df(df, col_truth + '_smoothed', col_pred, inplace=True, **kwargs)
        self.loss.compute_df(df, col_truth + '_smoothed', col_pred + '_norm', t_max=t_max, inplace=True, **kwargs)

        # find when data is available
        available_idx = df.index[~df[col_truth].isnull()]
        
        # find relevant time bounds
        if t_min is None:
            t_min = available_idx[0]
        if t_max is None:
            t_max = available_idx[-1]
        idx = df.loc[t_min:t_max].index

        # build results dictionary
        results = {
            'idx': df.loc[idx].index.values,
            'y_true': df.loc[idx, col_truth].values,
            'y_pred': df.loc[idx, col_pred].values,
            'y_true_smooth': df.loc[idx, col_truth + '_smoothed'].values,
            'y_pred_norm': df.loc[idx, col_pred + '_norm'].values,
            'y_pred_reg': df.loc[idx, col_pred + '_reg'].values,
            'y_pred_loss': df.loc[idx, col_pred + '_norm_loss'].values,
            'score': df.loc[idx, col_pred + '_norm_loss'].mean()
            }
        return results
    