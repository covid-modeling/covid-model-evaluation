from abc import ABC, abstractmethod
import numpy as np
import scipy.stats

class stager(ABC):
    @abstractmethod
    def test(self, y_true):
        """
        Tests whether an outbreak has reached or left a certain stage.

        Parameters:
        -----------
        y_true : array
            Reported cases values.

        Returns:
        --------
        progressed : boolean
            True iff the outbreak is judged to have progressed to a certain stage.
        p_value : float
            One sided p value for the hypothesis test whose null hypothesis is that the outbreak has not progressed
        """
        pass

class fit_and_compare_stager(stager):
    """
    A group of stagers that fit a function shape and test whether the true values crossed that function line
    suspiciously rarely.
    """

    def __init__(self, significance_level=.05):
        self.significance_level = significance_level

    def test(self, y_true):
        y_true = np.asarray(y_true)

        # remove left trail of zeros
        y_true = np.trim_zeros(y_true, trim='f')
        n_obs = np.shape(y_true)[0]

        if n_obs <= 2:
            # can't say anything
            return False, 1

        y_fit = np.round(self._fit(y_true))

        n_sign_changes = self._count_sign_changes(y_true - y_fit)

        # there can only be n-1 sign changes, so binom(n-1)
        p_value = scipy.stats.binom(n_obs-1, .5).cdf(n_sign_changes)

        return p_value < self.significance_level, p_value

    def _count_sign_changes(self, seq):
        """
        Counts the number of times the sign of the sequence changes.
        (Zeros will count as changes compared to the previous value, but not the subsequent value.)
        """
        seq = np.asarray(seq)
        n_zeros = np.sum(seq == 0)
        signs = np.sign(seq[np.nonzero(seq)])
        if signs.shape[0] > 0:
            n_sign_changes = np.sum(np.not_equal(signs[:-1], signs[1:]))
        else:
            n_sign_changes = 0
        return n_sign_changes + n_zeros

    @abstractmethod
    def _fit(self, y_true):
        """
        Fits a function to an array

        Parameters:
        -----------
        y_true : array
            Empirical values.

        Returns:
        --------
        y_fit : array
            Fitted values
        """
        pass

class exponential_stager(fit_and_compare_stager):
    def _fit(self, y_true):
        n_obs = y_true.shape[0]
        y_log = np.log(np.maximum(y_true, 1))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(range(n_obs), y_log)
        y_fit_log = range(n_obs) * slope + intercept
        y_fit = np.exp(y_fit_log)
        return y_fit
