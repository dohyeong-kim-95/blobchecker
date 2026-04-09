"""
lse/lse.py
Level-Set Estimation via sequential sampling.

Classes
-------
BaseEstimator   abstract interface
StraddleGPR     GP Regression + Straddle acquisition (Bryan et al. 2005)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


Oracle = Callable[[int, int], int]   # f(row, col) -> {0, 1}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEstimator(ABC):
    """Interface for level-set estimators."""

    @abstractmethod
    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
    ) -> np.ndarray:
        """Query oracle sequentially and return predicted binary map.

        Parameters
        ----------
        oracle    : deterministic f(row, col) -> {0, 1}
        grid_size : (H, W)
        budget    : total oracle calls; defaults to 15 % of H*W

        Returns
        -------
        (H, W) uint8 binary map
        """


# ---------------------------------------------------------------------------
# Straddle GPR
# ---------------------------------------------------------------------------

class StraddleGPR(BaseEstimator):
    """GP Regression with the Straddle acquisition function.

    Straddle heuristic (Bryan et al. 2005):
        score(x) = |mu(x) - 0.5| - kappa * sigma(x)
    The next sample is argmin of this score: points near the 0.5 decision
    boundary that also have high uncertainty are preferred.

    The GP is refit every `refit_interval` steps.  Between refits the
    acquisition ranking from the previous prediction is consumed in order
    (greedy batch), so the total number of GP solves is budget/refit_interval.
    """

    def __init__(
        self,
        kappa: float = 1.5,
        refit_interval: int = 50,
        n_init: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        kappa           : exploration weight in Straddle score
        refit_interval  : re-solve the GP every this many steps
        n_init          : initial random samples; defaults to refit_interval
        """
        self.kappa = kappa
        self.refit_interval = refit_interval
        self.n_init = n_init

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
    ) -> np.ndarray:
        H, W = grid_size
        total = H * W
        if budget is None:
            budget = int(0.15 * total)
        n_init = self.n_init if self.n_init is not None else self.refit_interval

        # Pixel coordinates normalised independently per axis → [0,1]²
        rows, cols = np.mgrid[0:H, 0:W]
        all_rc = np.column_stack([rows.ravel(), cols.ravel()])   # (N, 2)
        coords = all_rc / np.array([max(H - 1, 1), max(W - 1, 1)], dtype=float)

        # ---- initial random space-filling samples ----
        rng = np.random.default_rng(0)
        n_init = min(n_init, budget)
        init_idx = rng.choice(total, size=n_init, replace=False)

        sampled = np.zeros(total, dtype=bool)
        sampled[init_idx] = True
        X = coords[init_idx]
        y = np.array([oracle(all_rc[i, 0], all_rc[i, 1]) for i in init_idx],
                     dtype=float)

        # ---- GP setup ----
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
            * RBF(length_scale=0.15, length_scale_bounds=(0.02, 0.8))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=1,
            normalize_y=True,
        )
        gp.fit(X, y)
        n_sampled = n_init

        # ---- adaptive loop ----
        while n_sampled < budget:
            unsampled = np.where(~sampled)[0]
            if len(unsampled) == 0:
                break

            mu, sigma = gp.predict(coords[unsampled], return_std=True)
            score = np.abs(mu - 0.5) - self.kappa * sigma

            batch = min(self.refit_interval, budget - n_sampled)
            order = np.argsort(score)[:batch]

            for j in order:
                idx = unsampled[j]
                val = oracle(all_rc[idx, 0], all_rc[idx, 1])
                sampled[idx] = True
                X = np.vstack([X, coords[idx]])
                y = np.append(y, float(val))

            n_sampled += len(order)
            gp.fit(X, y)

        # ---- final prediction ----
        mu_all = gp.predict(coords)
        return (mu_all >= 0.5).astype(np.uint8).reshape(H, W)
