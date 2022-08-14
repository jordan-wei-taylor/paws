import numpy as np
from   scipy.stats import multivariate_normal, norm

class MultiVariateGaussian():
    """
    Multi-Variate Gaussian Model
    """
    def __init__(self, features, mask = None):
        self._features  = features
        self._mask      = mask
        self._nfeatures = False

    def set_mask(self, mask):
        """ sets a mask to indicate which features to use """
        self._mask = mask

    @property
    def features(self):
        return self._features if not self._nfeatures else self._features[self.mask]

    @property
    def mask(self):
        if self._nfeatures:
            return self._mask if isinstance(self._mask, (list, tuple, np.ndarray)) else np.ones(self._nfeatures, dtype = bool)
        assert Exception()


    def fit(self, data):
        """ fits a multi-variate gaussian model to a given dataset assuming observations are along the first dimension """
        self._nfeatures = data.shape[-1]

        self.mean       = data[:,self.mask].mean(axis = 0)
        self.cov        = np.cov(data[:,self.mask], rowvar = False)
        self.dist       = multivariate_normal(self.mean, self.cov, allow_singular = True)

        # precompute normalising constant
        s, _            = np.linalg.eigh(self.cov)
        mask            = s > 1e-8 # numerical stability
        constant        = np.zeros_like(mask, dtype = np.float64)
        constant[mask] += np.log(2 * np.pi * s[mask])

        self.constant   = constant / 2

        # precompute pseudo-inverse
        self._pinv      = np.linalg.pinv(self.cov)        

        # store original data
        self.data       = data

        return self

    def nll(self, *datasets):
        """ computes the negative log likelihood for each sample for each dataset """
        # shape is (n_datasets, n_samples)
        ret = np.array([self.dist.logpdf(data[:,self.mask]) for data in datasets]) # log likelihood
        return -ret                                                                # negative log likelihood

    def nll_by_feature(self, dataset):
        """ computes the negative log likelihood by feature for each sample for a dataset """
        ret     = []
        for data in dataset:
            # 0.5 * diag( (m - x) @ (m - x).T @ C^-1 )
            ret.append(np.diagonal((data[self.mask] - self.mean).reshape(-1, 1) @ (data[self.mask] - self.mean).reshape(1, -1) @ np.linalg.pinv(self.cov)) / 2)
        return ret + self.constant

    def ll_ratio_by_feature(self, *datasets):
        """ computes the log liklihood ratio by feature for each sample for each dataset """
        return np.array([np.minimum(self.nll_by_feature(self.data) - self.nll_by_feature(dataset), 0) for dataset in datasets])

    def l_ratio_by_feature(self, *datasets):
        """ computes the likihood ratio by featyre for eacg sample for each dataset """
        return np.exp(self.ll_ratio_by_feature(*datasets))
