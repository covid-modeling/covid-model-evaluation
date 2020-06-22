from abc import ABC, abstractmethod
import numpy as np
import scipy.signal

class smoother(ABC):
    @abstractmethod
    def smooth(self, y_true):
        """
        Smooths a 1-D array of values.

        Parameters:
        -----------
        y_truth : array
            Actual values.

        Returns:
        --------
        y_smoothed : array
            Smoothed values.
        """
        pass

    def smooth_df(self, df, col, inplace=False, **kwargs):
        """
        Similar to `smooth` method but accepts a DataFrame with a DatetimeIndex and handles NaNs. 
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame with a DatetimeIndex.
        col : str
            The column containing the vallues to be smoothed.
        inplace : bool
            If True updates `df` by storing results in column `$(col)_smoothed`.

        Returns
        -------
        A DataFrame with a `{$col}_smoothed` column if inplace=False, None otherwise.
        """
        # don't modify input dataframe unless specified
        if inplace:
            df.sort_index(ascending=True, inplace=True)
        else :
            df = df.sort_index(ascending=True)
        
        # find when data is available
        idx = df.index[~df[col].isnull()]
        
        # perform smoothing
        df.loc[idx, col + '_smoothed'] = self.smooth(df.loc[idx, col].values, **kwargs)
        
        if inplace:
            return None
        else:
            return df

class geometric(smoother):
    """
    A smoothing function that computes the geometric average of the current day with neighboring days,
    possibly with unequal, but symmetrical weights.

    Parameters
    ----------
        weights : array > 0
        The length of the array is the number of days in the future and past that will be used in smoothing.
        The weights are normalised so they sum to 1. If unspecified defaults to [.5, .3, .2].
    """

    def __init__(self, weights=None):
        # set default values
        if weights is None:
            weights = [.5, .3, .2]

        # check validity
        weights = np.asarray(weights)
        if weights.ndim != 1 or np.any(weights < 0) or not np.all(weights):
            raise ValueError("Weights must be non-negative values summing to > 0.")

        # normalise sum to 1
        self.weights = weights / np.sum(weights)

    def smooth(self, y_true):
        unrounded_series = self.smooth_continuous(y_true)
        return np.asarray([self._round(x) for x in unrounded_series])

    def _round(self, x):
        """
        Rounding function.

        Currently built in `round`, but may later be swapped out for one
        that does more appropriate rounding of small numbers with exponential disitribution
        """
        return round(x)

    def smooth_continuous(self, y_true):
        y_true = np.asarray(y_true)
        y_true_p1 = y_true + 1
        y_smoothed_p1 = np.ones(np.shape(y_true)[0])

        for i in range(len(y_true)):
            product = 1
            exponent = 0
            number_of_usable_weights_at_idx = min(np.shape(self.weights)[0], i+1, np.shape(y_true)[0]-i)
            for j in range(i-number_of_usable_weights_at_idx+1, i+number_of_usable_weights_at_idx):
                idx_in_weights = np.abs(j-i)
                current_exponent = self.weights[idx_in_weights]
                product *= np.power(y_true_p1[j], current_exponent)
                exponent += current_exponent
            y_smoothed_p1[i] = np.power(product, 1/exponent)

        y_smoothed = y_smoothed_p1 - 1
        return y_smoothed

class missed_cases(smoother):
    """
    A smoothing function that attempts to redistribute reported cases by assuming that some are reported on a later
    day. The aim is to attempt to preserve the cumulative count while reducing jerk in the data.
    
    This is geared towards administrative subunits missing reporting deadlines and similar effects.

    Parameters
    ----------
    cost_missing : float, optional
        Cost of reporting a single case one day later.
    cost_der1 : float, optional
        Cost of changing the number of daily new cases.
    cost_der2 : float, optional
        Cost of changing the rate at which new daily cases change.
    keep_cumul : boolean, optional
        If True forces the cumulative count to be the same.
    """

    def __init__(self, cost_missing=.1, cost_der1=10, cost_der2=1, keep_cumul=False):
        self.cost_missing = cost_missing
        self.cost_der = [cost_der1, cost_der2]
        self.keep_cumul = keep_cumul
        
        # since this smoother can be computationally expensive, we want to avoid recomputation on repeat calls
        self.cached_input = None
        self.cached_output = None

    def smooth(self, y_apparent, debug=False):
        if np.array_equal(self.cached_input, y_apparent):
            return self.cached_output
        self.cached_input = y_apparent
        y_apparent = np.asarray(y_apparent)
        n = y_apparent.shape[0]
        
        cases_missed = np.zeros(n, dtype=int)
        old_loss = self._loss(y_apparent, cases_missed)
        found_improvement = True
        n_max = n-1 if self.keep_cumul else n
        while found_improvement:
            found_improvement = False
            for i in range(n_max):
                for j in range(1,10):
                    if (j + i > n_max):
                        continue

                    # add 1 missed case for the j following days
                    suggestion = np.copy(cases_missed)
                    suggestion[i:i+j] += 1

                    # set number of missed cases on last day to 0 to preserve overall number of cases
                    if self.keep_cumul:
                        suggestion[-1] = 0

                    # compute corresponding loss
                    new_loss = self._loss(y_apparent, suggestion)

                    # accept suggestion if improvement
                    if new_loss < old_loss:
                        cases_missed = suggestion
                        old_loss = new_loss
                        found_improvement = True

        if debug:
            print(np.asarray([y_apparent, cases_missed, self._move_cases(y_apparent, cases_missed)]))
        self.cached_output = self._move_cases(y_apparent, cases_missed)
        return self.cached_output

    def _move_cases(self, y_apparent, cases_missed):
        # add cases missed for the day
        y_true = y_apparent + cases_missed
        
        # remove cases missed for the previous day
        y_true[1:] -= cases_missed[:-1]
        
        return y_true

    def _loss(self, y_apparent, cases_missed):
        # move cases
        y_true = self._move_cases(y_apparent, cases_missed)

        # handle impossibility to have more missed cases than the number of reported cases the following day
        if np.any(y_true < 0):
            return np.Inf
        
        # compute cost associated to new series
        loss_missed = np.sum(cases_missed) * self.cost_missing
        loss_derivatives = 0
        for i, cost in enumerate(self.cost_der):
            loss_derivatives += np.sum(np.abs(np.diff(y_true, 1+i))) * cost

        return loss_missed + loss_derivatives

class gaussian(smoother):
    """
    A low-pass Gaussian filter that aims at removing high-frequency noise, e.g. due to predictions being
    stochastic.

    Parameters
    ----------
    sigma : float, optional
        standard deviation of the Gaussian, in days.
    """

    def __init__(self, sigma=3):
        self.sigma = sigma

    def smooth(self, y_apparent):
        y_apparent = np.asarray(y_apparent)
        return np.round(scipy.ndimage.gaussian_filter1d(y_apparent, sigma=self.sigma))

class identity(smoother):
    """
    Does not perform any smoothing, provided as a convenience class only.
    """
    
    def smooth(self, y_true):
        return y_true
