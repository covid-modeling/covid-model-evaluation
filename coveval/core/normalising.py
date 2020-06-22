from abc import ABC, abstractmethod
import numpy as np
import scipy.stats

class normaliser(ABC):
    @abstractmethod
    def normalise(self, y_true, y_pred, *args, **kwargs):
        """
        Normalises predicted values, e.g. to account for things like:
        "Had I known at day x that so many people had died already, I'd have updated my predictions"

        Here, the values of y_norm[0 ... n+1] should only depend on y_true[0 ... n]

        Parameters:
        -----------
        y_true : array
            Reported values.
        y_pred : array
            Predicted values.

        Returns:
        --------
        y_norm : array
            Series of normalised predictions.
        y_reg : array
            Information about the normalisation applied at each timepoint.
        """
        pass
    
    def normalise_df(self, df, col_truth, col_pred, inplace=False):
        """
        Similar to `normalise()` method but accepts a Dataframe with a DatetimeIndex and handles NaNs. 
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame with a DatetimeIndex.
        col_truth : str
            Reported values.
        col_pred : str
            Predicted values.
        inplace : bool, optional
            If True, updates `df` by storing results in columns `${col_pred}_norm` and `${col_pred}_reg`.

        Returns
        -------
        A DataFrame with `{$col_pred}_norm` and `${col_pred}_reg` columns if inplace=False, None otherwise.
        """
        # don't modify input dataframe unless specified
        if inplace:
            df.sort_index(ascending=True, inplace=True)
        else :
            df = df.sort_index(ascending=True)
        
        # find when data is available
        idx = df.index[~df[col_truth].isnull()]
        
        # perform normalisation
        y_norm, y_reg = self.normalise(df.loc[idx, col_truth].values, df.loc[idx[0]:, col_pred].values)
        df.loc[idx, col_pred + '_norm'] = y_norm
        df.loc[idx, col_pred + '_reg'] = y_reg
        
        if inplace:
            return None
        else:
            return df

    def _slice(self, values, size=1, step=1):
        """
        Slice a given array into sub-array by applying a sliding window with specified width and step.

        Parameters
        ----------
        values : array
            A series of values.
        size : int, optional
            The width of the sliding window.
        step : int, optional
            The step between each window.

        Returns
        -------
        array of array
            An array containing sub-arrays of `values`.
        """
        if size > values.size:
            raise ValueError("Invalid size.")
        n = int((values.size + step - size)/step)
        index = np.arange(size).reshape(1,-1) + step * np.arange(n).reshape(-1,1)
        return values[index]

class dynamic_offset(normaliser):
    """
    Normalise a series of predicted values by shifting its timeline so that it best matches a specified number
    of previously observed values.
    """
    def _mle_poisson(self, lambdas, values, precision=7):
        """
        Detemine which series of Poisson distributions has the maximum likelihood of observing a series of values
        of same length.

        Parameters
        ----------
        lambdas : array ofshape (m,n)
            Array of parameters of n Poisson distributions.
        values : array of shape (n,) or (n,1)
            Corresponding set of values to observe.
        precision : int, optional
            Precision with which to compare log-likelihood values.

        Returns
        -------
        int
            Index `i` such that lambdas[i,:] are the lambdas with maximum likelihood of observing `values`.
        """
        _lle = np.sum(scipy.stats.poisson(lambdas).logpmf(values), axis=1)
        _lle = np.round(_lle, precision)
        return np.nonzero(_lle == np.max(_lle))[0]

    def normalise(self, y_true, y_pred, size=2, window=30, verbose=False, *args, **kwargs):
        """
        Adjust timeline of values of `y_pred` so that the likelihood of the `size` most recent observations is
        maximised.

        Parameters
        ----------
        y_true : array
            Reported values.
        y_pred : array
            Predicted values.
        size : int, optional
            The number of previous values to take into account for mle.
        window : int, optional
            Maximum number of `y_pred` values that can be shifted forward in one go.
        verbose : bool, optional
            If True print info during the normalisation process.

        Returns
        -------
        y_norm : array
            A normalised series of predicted values. Same size as `y_true`.
        y_offset : array
            Series of the corresponding offsets w.r.t. the original predicted values. Same size as `y_true`.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be arrays of dimension 1.")
        if y_true.shape[0] < size or y_pred.shape[0] < size:
            raise ValueError("size must be smaller than the number of elements in y_true and y_pred.")

        n = y_true.shape[0]
        y_norm = np.zeros(n)
        y_offset = np.zeros(n)

        y_norm[0:size] = y_pred[0:size]
        i_match = size - 1    
        for i in range(size, n):
            # previous n observed values
            last_obs = y_true[i-size:i]

            # find times in current window with maximum loglikelihood for last n observed values
            lambdas = self._slice(y_pred[i_match-(size-1):i_match+window], size=size)
            i_rel_candidates = self._mle_poisson(lambdas, last_obs)

            # select most suitable time as the one minimising gap with actual time
            i_rel = i_rel_candidates[np.argmin(np.abs(i_rel_candidates - ((i-1)-i_match)))]    

            # record next prediction and corresponding offset
            i_match = i_match + i_rel
            if i_match == y_pred.shape[0] - 1:
                raise ValueError(f"y_pred must include enough predictions (consumed all {y_pred.shape[0]} " + \
                                 f"provided prediction values fitting {i} of {n} true values)")
            y_norm[i] = y_pred[i_match + 1]
            y_offset[i] = i_match + 1 - i

            if verbose:
                print(f'i={i}, i_match={i_match}, y_prev={y_true[i-1]}, y_match={y_pred[i_match]}, ' +
                      f'i_offset={y_offset[i]}, y_true={y_true[i]}, y_pred={y_norm[i]}')

        return y_norm, y_offset

class dynamic_scaling(normaliser):
    """
    A normaliser that performs dynamic scaling, i.e. modifies a model's prediction according to how the model over-
    or underperformed over the last few days.

    Parameters
    ----------
    weights : array > 0
        The length of the array is the number of days in the past that will be used.
        The last entry corresponds to the day _closest_ to now.
        Thus, they should normally be weakly increasing -- however, this is not enforced.
        The weights are normalised so they sum to 1. If unspecified defaults to [.2, .3, .5].
    """
    def __init__(self, weights=None):
        # set default values
        if weights is None:
            weights = [.2, .3, .5]

        # check validity
        weights = np.asarray(weights)
        if weights.ndim != 1 or np.any(weights <= 0):
            raise ValueError("Weights must be strictly positive.")

        # normalise sum to 1
        self.weights = weights / np.sum(weights)

    def normalise(self, y_true, y_pred, *args, **kwargs):
        """
        Normalises predicted values so that each value is scaled by the factor corresponding to how the model
        over- or underperformed over the last few days for the period when this information is available.

        Parameters:
        -----------
        y_true: array
            Reported values.
        y_pred : array
            Predicted values.

        Returns:
        --------
        y_norm : array
            Series of normalised predictions.
        y_scale : array
            Series of scaling factor applied to predictions.
        """

        # offset everything by 1
        y_true = np.asarray(y_true) + 1
        y_pred = np.asarray(y_pred) + 1

        # validate inputs
        n_true = y_true.shape[0]
        window_size = self.weights.shape[0]
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be arrays of dimension 1.")
        if n_true < window_size:
            raise ValueError("too few elements in y_true.")
        if y_pred.shape[0] < window_size or y_pred.shape[0] < y_true.shape[0]:
            raise ValueError("too few elements in y_pred.")

        # initialise outputs
        y_scale = np.zeros(n_true)
        y_scale[:window_size] = 1
        y_norm = np.zeros(n_true)
        y_norm[:window_size] = y_pred[:window_size]

        # normalise predictions
        for i in range(window_size, n_true):
            # determine scaling factor
            expected = np.sum(self.weights * y_pred[i-window_size:i])
            actual = np.sum(self.weights * y_true[i-window_size:i])
            y_scale[i] = actual / expected
            
            # apply it to current prediction
            y_norm[i] = y_pred[i] * y_scale[i]

        return np.maximum(0, np.round(y_norm) - 1), y_scale

class identity(normaliser):
    """
    Does not perform any normalisation, provided as a convenience class only.
    """

    def normalise(self, y_true, y_pred, *args, **kwargs):
        return y_pred, np.zeros(np.asarray(y_pred).shape[0])
