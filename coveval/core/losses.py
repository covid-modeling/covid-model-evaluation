from abc import ABC, abstractmethod
import scipy.stats
import numpy as np

class loss(ABC):
    def compute(self, y_true, y_pred):
        """
        Compute loss vector.

        Parameters
        ----------
        y_true : array
            True values.
        y_pred: array
            Predicted values.

        Returns
        -------
        y_loss : array
            Loss value at each timepoint.
        """
        # check validity of inputs
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("Inputs `y_true` and `y_pred` must be arrays of dimension 1.")
        if np.shape(y_pred)[0] != np.shape(y_true)[0]:
            raise ValueError("Inputs `y_true` and `y_pred` must contain the same number of elements.")

        # sum pointwise loss
        n = np.shape(y_pred)[0]
        y_loss = np.zeros(n)
        for i in range(n):
            y_loss[i] = self.compute_pointwise(y_true[i], y_pred[i])
        return y_loss

    def compute_df(self, df, col_truth, col_pred, t_min=None, t_max=None, inplace=False, **kwargs):
        """
        Similar to `compute()` method but accepts a Dataframe with a DatetimeIndex and handles NaNs. 
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame with a DatetimeIndex.
        col_truth : str
            The column containing the reported data.
        col_pred : str
            The column containing the corresponding predictions.
        inplace : bool, optional
            If True updates `df` by storing results in column `${col_pred}_loss`.

        Returns
        -------
        A DataFrame with a `${col_pred}_loss` column if inplace=False, None otherwise.
        """
        # don't modify input dataframe unless specified
        if inplace:
            df.sort_index(ascending=True, inplace=True)
        else :
            df = df.sort_index(ascending=True)
        
        # find when data is available
        idx = df.index[~df[col_truth].isnull()]
                        
        # compute loss
        df.loc[idx, col_pred + '_loss'] = self.compute(df.loc[idx, col_truth].values,
                                                       df.loc[idx, col_pred].values, **kwargs)
        if inplace:
            return None
        else:
            return df

    @abstractmethod
    def compute_pointwise(self, y_true, y_pred):
        """
        Gives the loss for a single timepoint.

        Parameters
        ----------
        y_true : scalar
            True value.
        y_pred: scalar
            Predicted value.

        Returns
        -------
        y_loss : scalar
        """

class poisson(loss):
    def compute_pointwise(self, y_true, y_pred):
        # a Poisson dist with lambda = 0 has an infinite loss for any prediction other than 0, so we want to avoid
        # that without modifying our input data
        loss_pred = -scipy.stats.poisson(np.maximum(y_pred, 1)).logpmf(y_true)
        loss_truth = -scipy.stats.poisson(np.maximum(y_true, 1)).logpmf(y_true)
        loss_total = loss_pred - loss_truth
        return loss_total

class normal(loss):
    """
    Initializes a normal loss where the standard deviation associated to each mean value `mu` is defined as:
    ```
    sigma_fac * np.sqrt(mu +1)
    ```
    where `sigma_fac` is a constant factor.
    
    Parameters
    ----------
    sigma_fac : scalar
        A factor by which to multiply np.sqrt(mu+1) to obtain the standard deviation of the distribution.
    """
    def __init__(self, sigma_fac=1):
        self.sigma_fac = sigma_fac
        super().__init__() 

    def compute_pointwise(self, y_true, y_pred):
        return -scipy.stats.norm(y_pred, self.sigma_fac * np.sqrt(y_pred + 1)).logpdf(y_true)

class normal_scaled(loss):
    """
    Initializes a normal loss where the standard deviation associated to each mean value `mu` is defined as:
    ```
    pdf(mean*(1+/-delta_pc)) / pdf(mean) = rel_value
    ```
    where `delta_pc` and `rel_value` are constant.

    Parameters
    ----------
    delta_pc : scalar
        Absolute distance to mean mu in %.
    rel_value : scalar
        Relative value of the pdf desired for x = mean * (1 +/- delta_pc)
    """
    def __init__(self, delta_pc=0.5, rel_value=0.5):
        self.delta_pc = delta_pc
        self.rel_value = rel_value
        super().__init__() 

    def _infer_sigma(self, mu):
        return mu * self.delta_pc / np.sqrt(-2*np.log(self.rel_value))
    
    def compute_pointwise(self, y_true, y_pred):
        y_pred = max(y_pred, 1)
        return -scipy.stats.norm(y_pred, self._infer_sigma(y_pred)).logpdf(y_true)
