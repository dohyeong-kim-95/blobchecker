"""
sparse_gp/sparse_gp.py
Level-Set Estimation with a Sparse GP (SGPR / FITC via GPyTorch).

Classes
-------
BaseEstimator       same abstract interface as lse/lse.py
SparseStraddleGPR   SGPR + Straddle acquisition, warm-start refits
"""

from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import gpytorch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Oracle       = Callable[[int, int], int]
CheckpointFn = Callable[[int, np.ndarray], None]


# ---------------------------------------------------------------------------
# Abstract base  (mirrors lse/lse.py)
# ---------------------------------------------------------------------------

class BaseEstimator(ABC):
    @abstractmethod
    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> np.ndarray:
        """Query oracle sequentially; return predicted (H,W) uint8 binary map."""


# ---------------------------------------------------------------------------
# GPyTorch SGPR model  (FITC via InducingPointKernel)
# ---------------------------------------------------------------------------

class _SGPRModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        inducing_points: torch.Tensor,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            ),
            inducing_points=inducing_points,
            likelihood=likelihood,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ---------------------------------------------------------------------------
# Sparse Straddle GPR
# ---------------------------------------------------------------------------

class SparseStraddleGPR(BaseEstimator):
    """GP Regression with FITC sparse approximation + Straddle acquisition.

    Straddle score (Bryan et al. 2005):
        score(x) = |mu(x) - 0.5| - kappa * sigma(x)
    argmin is queried next.

    The GP is refit every `refit_interval` steps using warm-starting:
    hyperparameters and inducing-point locations carry over and are
    fine-tuned with `n_warm_steps` gradient steps.  The initial fit uses
    `n_init_steps` steps.
    """

    def __init__(
        self,
        kappa: float = 1.5,
        refit_interval: int = 50,
        n_inducing: int = 200,
        n_init_steps: int = 100,
        n_warm_steps: int = 30,
        lr: float = 0.1,
        n_init: int | None = None,
    ) -> None:
        self.kappa = kappa
        self.refit_interval = refit_interval
        self.n_inducing = n_inducing
        self.n_init_steps = n_init_steps
        self.n_warm_steps = n_warm_steps
        self.lr = lr
        self.n_init = n_init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train(
        self,
        model: _SGPRModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        n_steps: int,
    ) -> None:
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        with gpytorch.settings.cholesky_jitter(1e-4):
            for _ in range(n_steps):
                opt.zero_grad()
                loss = -mll(model(model.train_inputs[0]), model.train_targets)
                loss.backward()
                opt.step()

    def _predict(
        self,
        model: _SGPRModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        coords: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu_flat, sigma_flat) for the given coordinates."""
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.cholesky_jitter(1e-4):
            pred = likelihood(model(coords))
            mu    = pred.mean.numpy()
            sigma = pred.stddev.numpy()
        return mu, sigma

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(
        self,
        oracle: Oracle,
        grid_size: tuple[int, int],
        budget: int | None = None,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> np.ndarray:
        H, W = grid_size
        total = H * W
        if budget is None:
            budget = int(0.15 * total)
        n_init = self.n_init if self.n_init is not None else self.refit_interval

        # ---- grid coordinates, normalised to [0,1]² ----
        rows, cols = np.mgrid[0:H, 0:W]
        all_rc   = np.column_stack([rows.ravel(), cols.ravel()])
        scale    = np.array([max(H - 1, 1), max(W - 1, 1)], dtype=np.float32)
        coords_np = (all_rc / scale).astype(np.float32)
        coords_t  = torch.from_numpy(coords_np)         # (N, 2)

        # ---- initial random samples ----
        rng    = np.random.default_rng(0)
        n_init = min(n_init, budget)
        idx    = rng.choice(total, size=n_init, replace=False)

        sampled = np.zeros(total, dtype=bool)
        sampled[idx] = True
        X_np = coords_np[idx]
        y_np = np.array([oracle(all_rc[i, 0], all_rc[i, 1]) for i in idx],
                        dtype=np.float32)

        X_t = torch.from_numpy(X_np)
        y_t = torch.from_numpy(y_np)

        # ---- build initial model ----
        n_ind     = min(self.n_inducing, n_init)
        inducing  = X_t[:n_ind].clone()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model      = _SGPRModel(X_t, y_t, likelihood, inducing)

        self._train(model, likelihood, self.n_init_steps)
        n_sampled = n_init

        mu0, _ = self._predict(model, likelihood, coords_t)
        if checkpoint_fn is not None:
            checkpoint_fn(n_sampled, (mu0 >= 0.5).astype(np.uint8).reshape(H, W))

        # ---- adaptive loop ----
        while n_sampled < budget:
            unsampled = np.where(~sampled)[0]
            if len(unsampled) == 0:
                break

            mu, sigma = self._predict(model, likelihood, coords_t[unsampled])
            score = np.abs(mu - 0.5) - self.kappa * sigma

            batch = min(self.refit_interval, budget - n_sampled)
            order = np.argsort(score)[:batch]

            new_x = np.empty((len(order), 2), dtype=np.float32)
            new_y = np.empty(len(order), dtype=np.float32)
            for k, j in enumerate(order):
                real_idx = unsampled[j]
                val = oracle(all_rc[real_idx, 0], all_rc[real_idx, 1])
                sampled[real_idx] = True
                new_x[k] = coords_np[real_idx]
                new_y[k] = float(val)
            X_np = np.vstack([X_np, new_x])
            y_np = np.concatenate([y_np, new_y])

            n_sampled += len(order)

            # Warm-start: update training data, fine-tune
            X_t = torch.from_numpy(X_np)   # float32
            y_t = torch.from_numpy(y_np)   # float32
            model.set_train_data(inputs=X_t, targets=y_t, strict=False)
            self._train(model, likelihood, self.n_warm_steps)

            mu_ck, _ = self._predict(model, likelihood, coords_t)
            if checkpoint_fn is not None:
                checkpoint_fn(n_sampled,
                              (mu_ck >= 0.5).astype(np.uint8).reshape(H, W))

        # ---- final prediction ----
        mu_f, _ = self._predict(model, likelihood, coords_t)
        return (mu_f >= 0.5).astype(np.uint8).reshape(H, W)
