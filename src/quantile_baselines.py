from __future__ import annotations
import numpy as np
from scipy import interpolate
from scipy.stats.mstats import mquantiles
from sklearn.model_selection import train_test_split


def _estim_dist(quantiles, percentiles, y_min, y_max, smooth_tails, tau):
    """ Estimate CDF from list of quantiles, with smoothing """
    noise = np.random.uniform(low=0.0, high=1e-05, size=(len(quantiles),))
    noise_monotone = np.sort(noise)
    quantiles = quantiles + noise_monotone

    def interp1d(x, y, a, b):
        return interpolate.interp1d(
            x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True
        )

    cdf = interp1d(quantiles, percentiles, 0.0, 1.0)
    inv_cdf = interp1d(percentiles, quantiles, y_min, y_max)
    if smooth_tails:
        quantiles_smooth = quantiles
        tau_lo = tau
        tau_hi = 1 - tau
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = np.where(percentiles < tau_lo)[0]
        idx_hi = np.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = np.linspace(quantiles[0], q_lo, num=len(idx_lo))
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = np.linspace(q_hi, quantiles[-1], num=len(idx_hi))
        cdf = interp1d(quantiles_smooth, percentiles, 0.0, 1.0)
        inv_cdf = interp1d(percentiles, quantiles_smooth, y_min, y_max)
    breaks = np.linspace(y_min, y_max, num=1000, endpoint=True)
    cdf_hat = cdf(breaks)
    f_hat = np.diff(cdf_hat)
    f_hat = (f_hat + 1e-06) / np.sum(f_hat + 1e-06)
    cdf_hat = np.concatenate([[0], np.cumsum(f_hat)])
    cdf = interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = interp1d(cdf_hat, breaks, y_min, y_max)
    return (cdf, inv_cdf)


def find_nearest(a, a0):
    """Index of element in nd array `a` closest to the scalar value `a0`"""
    idx = np.abs(a - a0).argmin()
    return idx


class QR_errfun:
    """Calculates conformalized quantile regression error.
  Conformity scores:
  .. math::
  max{\\hat{q}_low - y, y - \\hat{q}_high}
  """

    def __init__(self):
        super(QR_errfun, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, alpha):
        q = np.quantile(
            nc, np.minimum(1.0, (1.0 - alpha) * (nc.shape[0] + 1.0) / nc.shape[0])
        )
        return np.vstack([q, q])


class DistSplit:
    """
  Method from "Flexible distribution-free conditional predictive bandsusing density estimators"
  """

    def __init__(self, bbox=None, ymin=-1, ymax=1):
        if bbox is not None:
            self.init_bbox(bbox)
        self.ymin = ymin
        self.ymax = ymax

    def init_bbox(self, bbox):
        self.bbox = bbox

    def fit(self, X, Y):
        self.bbox.fit(X, Y)

    def calibrate(self, X, Y, alpha, bbox=None, return_scores=False):
        self.alpha = alpha
        if bbox is not None:
            self.init_bbox(bbox)
        n2 = X.shape[0]
        quantiles = self.bbox.predict(X)
        percentiles = self.bbox.get_quantiles()
        scores = np.array([0.0] * n2)
        for i in range(n2):
            (cdf, inv_cdf) = _estim_dist(
                quantiles[i],
                percentiles,
                y_min=self.ymin,
                y_max=self.ymax,
                smooth_tails=True,
                tau=0.01,
            )
            scores[i] = cdf(Y[i])
        alpha_adjusted = 1 - (1 - alpha) * (1.0 + 1.0 / float(n2))
        self.t_lo = mquantiles(scores, prob=alpha_adjusted / 2)[0]
        self.t_up = mquantiles(scores, prob=1.0 - alpha_adjusted / 2)[0]

    def fit_calibrate(self, X, Y, alpha, bbox=None, random_state=2020, verbose=False):
        self.alpha = alpha
        if bbox is not None:
            self.init_bbox(bbox)
        (X_train, X_calib, Y_train, Y_calib) = train_test_split(
            X, Y, test_size=0.5, random_state=random_state
        )
        self.fit(X_train, Y_train)
        self.calibrate(X_calib, Y_calib, alpha)

    def predict(self, X):
        quantiles = self.bbox.predict(X)
        percentiles = self.bbox.get_quantiles()
        n = X.shape[0]
        pred = np.zeros((n, 2))
        for i in range(n):
            (cdf, inv_cdf) = _estim_dist(
                quantiles[i],
                percentiles,
                y_min=self.ymin,
                y_max=self.ymax,
                smooth_tails=True,
                tau=0.01,
            )
            pred[i, 0] = inv_cdf(self.t_lo)
            pred[i, 1] = inv_cdf(self.t_up)
        return pred


class DCP:
    """
  Method from "Distributional conformal prediction"
  """

    def __init__(self, bbox=None, ymin=-1, ymax=1):
        if bbox is not None:
            self.init_bbox(bbox)
        self.ymin = ymin
        self.ymax = ymax

    def init_bbox(self, bbox):
        self.bbox = bbox

    def fit(self, X, Y):
        self.bbox.fit(X, Y)

    def calibrate(self, X, Y, alpha, bbox=None, return_scores=False):
        self.alpha = alpha
        if bbox is not None:
            self.init_bbox(bbox)
        n2 = X.shape[0]
        quantiles = self.bbox.predict(X)
        percentiles = self.bbox.get_quantiles()
        cdf_values = np.array([0.0] * n2)
        for i in range(n2):
            (cdf, inv_cdf) = _estim_dist(
                quantiles[i],
                percentiles,
                y_min=self.ymin,
                y_max=self.ymax,
                smooth_tails=True,
                tau=0.01,
            )
            cdf_values[i] = cdf(Y[i])
        scores = np.abs(np.clip(cdf_values, 0, 1) - 1 / 2)
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        self.alpha_calibrated = 0.5 - mquantiles(scores, prob=level_adjusted)[0]

    def fit_calibrate(self, X, Y, alpha, bbox=None, random_state=2020, verbose=False):
        self.alpha = alpha
        if bbox is not None:
            self.init_bbox(bbox)
        (X_train, X_calib, Y_train, Y_calib) = train_test_split(
            X, Y, test_size=0.5, random_state=random_state
        )
        self.fit(X_train, Y_train)
        self.calibrate(X_calib, Y_calib, alpha)

    def predict(self, X):
        quantiles = self.bbox.predict(X)
        percentiles = self.bbox.get_quantiles()
        n = X.shape[0]
        pred = np.zeros((n, 2))
        for i in range(n):
            (cdf, inv_cdf) = _estim_dist(
                quantiles[i],
                percentiles,
                y_min=self.ymin,
                y_max=self.ymax,
                smooth_tails=True,
                tau=0.01,
            )
            pred[i, 0] = inv_cdf(self.alpha_calibrated)
            pred[i, 1] = inv_cdf(1 - self.alpha_calibrated)
        return pred


class CQR:
    """
  Classical CQR
  """

    def __init__(self, bbox=None):
        if bbox is not None:
            self.init_bbox(bbox)

    def init_bbox(self, bbox):
        self.bbox = bbox

    def fit(self, X, Y):
        self.bbox.fit(X, Y)

    def calibrate(self, X, Y, alpha, bbox=None, return_scores=False):
        self.alpha = alpha
        if bbox is not None:
            self.init_bbox(bbox)
        n2 = X.shape[0]
        pred = self.bbox.predict(X)
        quantiles = self.bbox.get_quantiles()
        idx_lower = find_nearest(quantiles, self.alpha / 2.0)
        idx_upper = find_nearest(quantiles, 1.0 - self.alpha / 2.0)
        pred = pred[:, [idx_lower, idx_upper]]
        scorer = QR_errfun()
        scores = scorer.apply(pred, Y)
        self.score_correction = scorer.apply_inverse(scores, alpha)
        print(
            "Calibrated score corrections: {:.3f}, {:.3f}".format(
                -self.score_correction[0, 0], self.score_correction[1, 0]
            )
        )

    def fit_calibrate(self, X, Y, alpha, bbox=None, random_state=2020, verbose=False):
        self.alpha = alpha
        if bbox is not None:
            self.init_bbox(bbox)
        (X_train, X_calib, Y_train, Y_calib) = train_test_split(
            X, Y, test_size=0.5, random_state=random_state
        )
        self.fit(X_train, Y_train)
        self.calibrate(X_calib, Y_calib, alpha)

    def predict(self, X):
        quantiles = self.bbox.get_quantiles()
        idx_lower = find_nearest(quantiles, self.alpha / 2.0)
        idx_upper = find_nearest(quantiles, 1.0 - self.alpha / 2.0)
        pred = self.bbox.predict(X)
        pred = pred[:, [idx_lower, idx_upper]]
        pred[:, 0] -= self.score_correction[0, 0]
        pred[:, 1] += self.score_correction[1, 0]
        return pred

    def predict_all(self, X):
        pred = self.bbox.predict(X)
        return pred


class CQR2:
    """
    CQR with inverse quantile scores
    """

    def __init__(self, bbox=None):
        if bbox is not None:
            self.init_bbox(bbox)

    def init_bbox(self, bbox):
        self.bbox = bbox
        quantiles = bbox.quantiles
        assert (np.diff(quantiles) >= 0).all()
        num_quantiles = len(quantiles)
        num_alpha = int(np.floor(num_quantiles / 2))
        assert num_alpha > 1
        quantiles_idx = np.arange(num_quantiles)
        qidx_low = quantiles_idx[0:num_alpha]
        self.qidx_low = -np.sort(-qidx_low)
        self.qidx_high = quantiles_idx[len(quantiles) - num_alpha : len(quantiles)]

    def fit(self, X, Y, bbox=None):
        if bbox is not None:
            self.init_bbox(bbox)
        self.bbox.fit(X.astype(np.float32), Y.astype(np.float32))

    def calibrate(self, X_calib, Y_calib, alpha, bbox=None, return_scores=False):
        if bbox is not None:
            self.init_bbox(bbox)
        pred = self.bbox.predict(X_calib)
        quantiles = self.bbox.get_quantiles()
        num_quantiles = len(quantiles)
        pred_low = pred[:, self.qidx_low]
        pred_high = pred[:, self.qidx_high]
        Y_c_mat = Y_calib.reshape((len(Y_calib), 1))
        covered = (Y_c_mat >= pred_low) * (Y_c_mat <= pred_high)
        covered = np.pad(
            covered, ((0, 0), (1, 1)), "constant", constant_values=(False, True)
        )
        scores = np.argmax(covered == True, axis=1)
        scores = scores - 1
        scores[np.where(scores < 0)] = 0
        n2 = X_calib.shape[0]
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        calibrated_idx = int(mquantiles(scores, prob=level_adjusted)[0])
        if calibrated_idx >= len(self.qidx_low):
            calibrated_idx = len(self.qidx_low) - 1
        self.calibrated_qidx_low = self.qidx_low[calibrated_idx]
        self.calibrated_qidx_high = self.qidx_high[calibrated_idx]
        pred_cqr = np.zeros((n2, 2))
        pred_cqr[:, 0] = pred[:, self.calibrated_qidx_low]
        pred_cqr[:, 1] = pred[:, self.calibrated_qidx_high]
        scorer_cqr = QR_errfun()
        scores_cqr = scorer_cqr.apply(pred_cqr, Y_calib)
        self.cqr_correction = scorer_cqr.apply_inverse(scores_cqr, alpha).flatten()
        q_star_low = quantiles[self.calibrated_qidx_low]
        q_star_high = quantiles[self.calibrated_qidx_high]
        print(
            "Calibrated quantiles (nominal level: {}): {:.3f},{:.3f}; CQR correction: {:.3f}".format(
                alpha, q_star_low, q_star_high, self.cqr_correction[0]
            )
        )
        scores_out = scores
        scores_out[scores_out == len(self.qidx_low)] = len(self.qidx_low) - 1
        scores_out = 1.0 - 2 * quantiles[self.qidx_low[scores_out]]
        if return_scores:
            return scores_out

    def fit_calibrate(
        self,
        X,
        Y,
        alpha,
        random_state=2020,
        bbox=None,
        verbose=False,
        return_scores=False,
    ):
        if bbox is not None:
            self.init_bbox(bbox)
        (X_train, X_calib, Y_train, Y_calib) = train_test_split(
            X, Y, test_size=0.5, random_state=random_state
        )
        self.fit(X_train, Y_train)
        scores = self.calibrate(X_calib, Y_calib, alpha)
        if return_scores:
            return scores

    def predict(self, X):
        pred = self.bbox.predict(X)
        pred_low = pred[:, self.calibrated_qidx_low] - self.cqr_correction[0]
        pred_high = pred[:, self.calibrated_qidx_high] + self.cqr_correction[1]
        return np.concatenate(
            (pred_low[:, np.newaxis], pred_high[:, np.newaxis]), axis=1
        ).squeeze()

    def predict_all(self, X):
        pred = self.bbox.predict(X)
        return pred
