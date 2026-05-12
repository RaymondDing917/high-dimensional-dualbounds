"""
Reproducible demo experiment for vector-valued discrete VarITE dual bounds.

Estimand:
    VarITE_sq = E[||Y(1) - Y(0)||_2^2] - ||E[Y(1) - Y(0)]||_2^2

Here Y(0), Y(1) are dim_y-dimensional binary vectors with full support
{0, 1}^dim_y. The default dim_y=10 gives support size 1024.

Examples:
    # Smoke run.
    python dual_bounds_single.py --n-list 100 --reps 1 --model-type oracle \
        --dual-max-units 10 --oracle-max-units 10

    # Unit tests.
    python dual_bounds_single.py test
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import unittest
import warnings
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

try:
    import ot
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise ImportError(
        "This experiment needs POT (`ot`) for exact discrete OT. "
        "Install it with `pip install POT`."
    ) from exc


# ---------------------------------------------------------------------------
# Support, costs, and numerics
# ---------------------------------------------------------------------------


def elapsed(t0: float) -> float:
    return float(np.around(time.time() - t0, 3))


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {value}")


def parse_ot_threads(value):
    if isinstance(value, int):
        return value
    value = str(value)
    if value == "max":
        return "max"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--ot-num-threads must be positive or 'max'")
    return parsed


def make_binary_support(dim_y: int) -> np.ndarray:
    """Return the full {0,1}^dim_y support as an array of shape (2^dim_y, dim_y)."""
    if dim_y <= 0:
        raise ValueError("dim_y must be positive")
    atoms = np.arange(2**dim_y, dtype=np.int64)
    shifts = np.arange(dim_y - 1, -1, -1, dtype=np.int64)
    return ((atoms[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def build_vector_cost_matrix(
    y0_vals: np.ndarray,
    y1_vals: np.ndarray,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """
    Build C[a,b] = ||y1_vals[b] - y0_vals[a]||_2^2.

    The OT/dual solver only sees this scalar cost matrix; it does not need to
    know that each atom is a vector.
    """
    if metric != "sqeuclidean":
        raise ValueError(f"Unsupported metric={metric}")
    y0_vals = np.asarray(y0_vals, dtype=np.float64)
    y1_vals = np.asarray(y1_vals, dtype=np.float64)
    if y0_vals.ndim != 2 or y1_vals.ndim != 2:
        raise ValueError("y0_vals and y1_vals must be two-dimensional")
    if y0_vals.shape[1] != y1_vals.shape[1]:
        raise ValueError("y0_vals and y1_vals must have the same vector dimension")
    diff = y1_vals[None, :, :] - y0_vals[:, None, :]
    return np.ascontiguousarray(np.sum(diff**2, axis=2), dtype=np.float64)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    validate_probability_matrix(probs, name="softmax_probs")
    return probs


def validate_probability_matrix(probs: np.ndarray, name: str, atol: float = 1e-8) -> None:
    if probs.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{name} contains NaN or inf")
    if np.any(probs < -atol):
        raise ValueError(f"{name} contains negative probabilities")
    rowsums = probs.sum(axis=1)
    if not np.allclose(rowsums, 1, atol=atol):
        maxerr = np.max(np.abs(rowsums - 1))
        raise ValueError(f"{name} rows do not sum to 1; max error={maxerr}")


def sample_atoms_from_probs(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    validate_probability_matrix(probs, name="sampling_probs")
    cumprobs = np.cumsum(probs, axis=1)
    draws = rng.random(probs.shape[0])
    return np.argmax(draws[:, None] <= cumprobs, axis=1).astype(np.int64)


def expected_y(probs: np.ndarray, support_y: np.ndarray) -> np.ndarray:
    return probs @ support_y.astype(np.float64)


def vector_varite_delta_method_se(
    sbetas: np.ndarray,
    skappa1s: np.ndarray,
    skappa0s: np.ndarray,
) -> tuple[float, float]:
    """
    Delta-method estimate and standard error for vector VarITE_sq.

    sbetas estimates E[||Y(1)-Y(0)||^2]. skappa1s/skappa0s estimate
    E[Y(1)] and E[Y(0)] componentwise.
    """
    sbetas = np.asarray(sbetas, dtype=np.float64)
    skappa1s = np.asarray(skappa1s, dtype=np.float64)
    skappa0s = np.asarray(skappa0s, dtype=np.float64)
    if skappa1s.shape != skappa0s.shape:
        raise ValueError("skappa1s and skappa0s must have the same shape")

    beta_hat = float(sbetas.mean())
    kappa1_hat = skappa1s.mean(axis=0)
    kappa0_hat = skappa0s.mean(axis=0)
    ate_hat = kappa1_hat - kappa0_hat
    theta_hat = beta_hat - float(ate_hat @ ate_hat)

    stacked = np.column_stack([sbetas, skappa1s, skappa0s])
    sigma = np.cov(stacked, rowvar=False)
    grad = np.concatenate([[1.0], -2 * ate_hat, 2 * ate_hat])
    se = float(np.sqrt(max(grad @ sigma @ grad / len(sbetas), 0.0)))
    return theta_hat, se


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


@dataclass
class BinaryVectorDGPData:
    X: np.ndarray
    W: np.ndarray
    Y: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    probs0_true: np.ndarray
    probs1_true: np.ndarray
    atom0: np.ndarray
    atom1: np.ndarray
    atom_obs: np.ndarray
    support_Y: np.ndarray
    propensity: float


def generate_binary_vector_dgp(
    n: int,
    p_x: int = 50,
    dim_y: int = 10,
    propensity: float = 0.3,
    alpha_scale: float = 0.5,
    beta_scale: float = 0.3,
    seed: int = 1,
) -> BinaryVectorDGPData:
    """
    Generate high-dimensional discrete potential outcomes.

    Conditional PMFs are full-support softmax models:
        logits_k[i,j] = alpha_k[j] + X_i^T beta_k[:,j]
    with beta coefficients scaled by beta_scale / sqrt(p_x).
    """
    rng = np.random.default_rng(seed)
    support_y = make_binary_support(dim_y)
    support_size = support_y.shape[0]

    X = rng.normal(size=(n, p_x))
    alpha0 = rng.normal(scale=alpha_scale, size=support_size)
    alpha1 = rng.normal(scale=alpha_scale, size=support_size)
    beta_sd = beta_scale / math.sqrt(p_x)
    beta0 = rng.normal(scale=beta_sd, size=(p_x, support_size))
    beta1 = rng.normal(scale=beta_sd, size=(p_x, support_size))

    probs0_true = stable_softmax(X @ beta0 + alpha0[None, :])
    probs1_true = stable_softmax(X @ beta1 + alpha1[None, :])

    atom0 = sample_atoms_from_probs(probs0_true, rng)
    atom1 = sample_atoms_from_probs(probs1_true, rng)
    Y0 = support_y[atom0].astype(np.int8)
    Y1 = support_y[atom1].astype(np.int8)
    W = rng.binomial(1, propensity, size=n).astype(np.int8)
    atom_obs = np.where(W == 1, atom1, atom0).astype(np.int64)
    Y = support_y[atom_obs].astype(np.int8)

    return BinaryVectorDGPData(
        X=X,
        W=W,
        Y=Y,
        Y0=Y0,
        Y1=Y1,
        probs0_true=probs0_true,
        probs1_true=probs1_true,
        atom0=atom0,
        atom1=atom1,
        atom_obs=atom_obs,
        support_Y=support_y,
        propensity=propensity,
    )


# ---------------------------------------------------------------------------
# Outcome PMF estimation
# ---------------------------------------------------------------------------


@dataclass
class PMFPrediction:
    probs: np.ndarray
    observed_unique_atoms: int
    warning_count: int


def _smooth_full_support_probs(probs: np.ndarray, smoothing_epsilon: float) -> np.ndarray:
    probs = probs + smoothing_epsilon
    probs = probs / probs.sum(axis=1, keepdims=True)
    validate_probability_matrix(probs, name="smoothed_probs", atol=1e-7)
    return probs


def fit_predict_multinomial_pmf(
    X_train: np.ndarray,
    atom_train: np.ndarray,
    X_eval: np.ndarray,
    support_size: int,
    smoothing_epsilon: float = 1e-6,
    logit_c: float = 1.0,
    logit_solver: str = "lbfgs",
    logit_max_iter: int = 200,
    seed: int = 1,
) -> PMFPrediction:
    """
    Fit P(Y=z_j | X, W=k) on one treatment arm and return full-support PMFs.

    Missing classes are assigned zero probability before smoothing.
    """
    classes = np.unique(atom_train)
    full_probs = np.zeros((X_eval.shape[0], support_size), dtype=np.float64)
    warning_count = 0

    if len(classes) == 0:
        full_probs[:] = 1.0 / support_size
        return PMFPrediction(
            probs=_smooth_full_support_probs(full_probs, smoothing_epsilon),
            observed_unique_atoms=0,
            warning_count=0,
        )

    if len(classes) == 1:
        full_probs[:, int(classes[0])] = 1.0
        return PMFPrediction(
            probs=_smooth_full_support_probs(full_probs, smoothing_epsilon),
            observed_unique_atoms=1,
            warning_count=0,
        )

    model = LogisticRegression(
        penalty="l2",
        C=logit_c,
        solver=logit_solver,
        max_iter=logit_max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, atom_train)
        warning_count = sum(issubclass(w.category, Warning) for w in caught)

    pred = model.predict_proba(X_eval)
    full_probs[:, model.classes_.astype(int)] = pred
    return PMFPrediction(
        probs=_smooth_full_support_probs(full_probs, smoothing_epsilon),
        observed_unique_atoms=len(classes),
        warning_count=warning_count,
    )


def fit_predict_two_arm_pmfs(
    data: BinaryVectorDGPData,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    smoothing_epsilon: float,
    logit_c: float,
    logit_solver: str,
    logit_max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    support_size = data.support_Y.shape[0]
    train0 = train_idx[data.W[train_idx] == 0]
    train1 = train_idx[data.W[train_idx] == 1]

    pred0 = fit_predict_multinomial_pmf(
        X_train=data.X[train0],
        atom_train=data.atom_obs[train0],
        X_eval=data.X[eval_idx],
        support_size=support_size,
        smoothing_epsilon=smoothing_epsilon,
        logit_c=logit_c,
        logit_solver=logit_solver,
        logit_max_iter=logit_max_iter,
        seed=seed,
    )
    pred1 = fit_predict_multinomial_pmf(
        X_train=data.X[train1],
        atom_train=data.atom_obs[train1],
        X_eval=data.X[eval_idx],
        support_size=support_size,
        smoothing_epsilon=smoothing_epsilon,
        logit_c=logit_c,
        logit_solver=logit_solver,
        logit_max_iter=logit_max_iter,
        seed=seed + 17,
    )
    diagnostics = {
        "observed_unique_atoms_arm0": pred0.observed_unique_atoms,
        "observed_unique_atoms_arm1": pred1.observed_unique_atoms,
        "model_warning_count": pred0.warning_count + pred1.warning_count,
    }
    return pred0.probs, pred1.probs, diagnostics


# ---------------------------------------------------------------------------
# Exact discrete OT duals and SE-adjusted estimator duals
# ---------------------------------------------------------------------------


SE_SOLVER = "CLARABEL"
CVXPY_SUCCESS_STATUSES = {"optimal", "optimal_inaccurate"}


@dataclass
class OTBoundPair:
    lower_value: float
    upper_value: float
    lower_u0: np.ndarray
    lower_u1: np.ndarray
    upper_u0: np.ndarray
    upper_u1: np.ndarray
    warning_count: int
    status: str


@dataclass
class OTDiagnostics:
    solve_count: int = 0
    cache_hits: int = 0
    runtime_seconds: float = 0.0


@dataclass
class SEDualSolution:
    value: float
    objective_value: float
    u0: np.ndarray
    u1: np.ndarray
    warning_count: int
    status: str


@dataclass
class SEDualBoundPair:
    lower_value: float
    upper_value: float
    lower_u0: np.ndarray
    lower_u1: np.ndarray
    upper_u0: np.ndarray
    upper_u1: np.ndarray
    warning_count: int
    status: str


def _normalize_prob_vector(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64).copy()
    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("probability vector has non-positive or non-finite mass")
    probs /= total
    return probs


def _ot_cache_key(
    probs0: np.ndarray,
    probs1: np.ndarray,
    cost: np.ndarray,
) -> tuple[object, ...]:
    return (
        id(cost),
        cost.shape,
        cost.dtype.str,
        np.ascontiguousarray(probs0).tobytes(),
        np.ascontiguousarray(probs1).tobytes(),
    )


def _maybe_print_ot_progress(
    label: str,
    row: int,
    total: int,
    progress_every: int,
    quiet: bool,
    diagnostics: OTDiagnostics | None,
) -> None:
    if quiet or progress_every <= 0:
        return
    done = row + 1
    if done != 1 and done != total and done % progress_every != 0:
        return
    diag_msg = ""
    if diagnostics is not None:
        diag_msg = (
            f" solves={diagnostics.solve_count}"
            f" cache_hits={diagnostics.cache_hits}"
            f" solver_runtime={diagnostics.runtime_seconds:.3f}s"
        )
    print(f"[solver] {label}: {done}/{total}{diag_msg}", flush=True)


def solve_ot_bound_pair(
    probs0: np.ndarray,
    probs1: np.ndarray,
    cost: np.ndarray,
    num_threads: int | str = 1,
    num_iter_max: int = 100000,
    cache: dict[tuple[object, ...], OTBoundPair] | None = None,
    diagnostics: OTDiagnostics | None = None,
) -> OTBoundPair:
    """
    Solve lower and upper first-term OT bounds and return dual potentials.

    Lower solves min_gamma <gamma, C>. Upper solves -min_gamma <gamma, -C>.
    POT returns minimization dual potentials u, v satisfying u[a]+v[b] <= C[a,b].
    For the upper bound, potentials are negated so they satisfy phi+psi >= C.
    """
    probs0 = _normalize_prob_vector(probs0)
    probs1 = _normalize_prob_vector(probs1)

    cache_key = None
    if cache is not None:
        cache_key = _ot_cache_key(probs0, probs1, cost)
        cached = cache.get(cache_key)
        if cached is not None:
            if diagnostics is not None:
                diagnostics.cache_hits += 1
            return cached

    t0 = time.time()
    lower_cost, lower_log = ot.emd2(
        probs0,
        probs1,
        cost,
        log=True,
        return_matrix=False,
        numItermax=num_iter_max,
        numThreads=num_threads,
        check_marginals=False,
    )
    upper_neg_cost, upper_log = ot.emd2(
        probs0,
        probs1,
        -cost,
        log=True,
        return_matrix=False,
        numItermax=num_iter_max,
        numThreads=num_threads,
        check_marginals=False,
    )
    if diagnostics is not None:
        diagnostics.solve_count += 1
        diagnostics.runtime_seconds += time.time() - t0

    warning_count = int(lower_log.get("warning") is not None)
    warning_count += int(upper_log.get("warning") is not None)
    status = "ok" if warning_count == 0 else "warning"

    result = OTBoundPair(
        lower_value=float(lower_cost),
        upper_value=float(-upper_neg_cost),
        lower_u0=np.asarray(lower_log["u"], dtype=np.float64),
        lower_u1=np.asarray(lower_log["v"], dtype=np.float64),
        upper_u0=-np.asarray(upper_log["u"], dtype=np.float64),
        upper_u1=-np.asarray(upper_log["v"], dtype=np.float64),
        warning_count=warning_count,
        status=status,
    )
    if cache is not None and cache_key is not None:
        cache[cache_key] = result
    return result


def _sqrt_psd_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    return evecs @ np.diag(np.sqrt(np.maximum(evals, 0.0))) @ evecs.T


def _solve_se_adjusted_lower_dual(
    probs0: np.ndarray,
    probs1: np.ndarray,
    cost: np.ndarray,
    pi: float,
    n: int,
    solver: str = SE_SOLVER,
) -> SEDualSolution:
    """
    Solve the SE-adjusted lower dual problem from generic.py.

    This maximizes E[nu0(Y0)] + E[nu1(Y1)] minus two estimated standard
    errors subject to the usual dual feasibility constraints.
    """
    probs0 = _normalize_prob_vector(probs0)
    probs1 = _normalize_prob_vector(probs1)
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != (len(probs0), len(probs1)):
        raise ValueError("cost shape must match probability vector lengths")
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must be strictly between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")

    nu0 = cp.Variable(len(probs0))
    nu1 = cp.Variable(len(probs1))
    nusum = cp.reshape(nu0, (len(probs0), 1), order="C") + cp.reshape(
        nu1, (1, len(probs1)), order="C"
    )
    linobj = probs0 @ nu0 + probs1 @ nu1

    pcat = np.concatenate([probs0, probs1])
    qmat = np.diag(np.concatenate([probs0 / (1.0 - pi), probs1 / pi])) - np.outer(
        pcat, pcat
    )
    sqrt_q = _sqrt_psd_matrix(qmat)
    stdev = cp.norm2(sqrt_q @ cp.hstack([nu0, nu1]))
    problem = cp.Problem(
        cp.Maximize(linobj - 2.0 * stdev / math.sqrt(n)),
        constraints=[nusum <= cost],
    )

    try:
        problem.solve(solver=solver)
    except Exception as exc:  # pragma: no cover - solver-environment dependent.
        warnings.warn(
            f"SE dual solver failed with {type(exc).__name__}: {exc}",
            RuntimeWarning,
        )
        return SEDualSolution(
            value=0.0,
            objective_value=0.0,
            u0=np.zeros(len(probs0)),
            u1=np.zeros(len(probs1)),
            warning_count=1,
            status="error",
        )

    if problem.status not in CVXPY_SUCCESS_STATUSES:
        warnings.warn(
            f"SE dual solver returned status {problem.status}; using zero potentials.",
            RuntimeWarning,
        )
        return SEDualSolution(
            value=0.0,
            objective_value=0.0,
            u0=np.zeros(len(probs0)),
            u1=np.zeros(len(probs1)),
            warning_count=1,
            status=str(problem.status),
        )

    u0 = np.asarray(nu0.value, dtype=np.float64).reshape(-1)
    u1 = np.asarray(nu1.value, dtype=np.float64).reshape(-1)
    if not (np.all(np.isfinite(u0)) and np.all(np.isfinite(u1))):
        warnings.warn("SE dual solver returned non-finite potentials.", RuntimeWarning)
        return SEDualSolution(
            value=0.0,
            objective_value=0.0,
            u0=np.zeros(len(probs0)),
            u1=np.zeros(len(probs1)),
            warning_count=1,
            status="nonfinite",
        )

    return SEDualSolution(
        value=float(probs0 @ u0 + probs1 @ u1),
        objective_value=float(problem.value),
        u0=u0,
        u1=u1,
        warning_count=0,
        status=str(problem.status),
    )


def solve_se_bound_pair(
    probs0: np.ndarray,
    probs1: np.ndarray,
    cost: np.ndarray,
    pi: float,
    n: int,
    solver: str = SE_SOLVER,
    diagnostics: OTDiagnostics | None = None,
) -> SEDualBoundPair:
    """
    Solve SE-adjusted lower/upper dual potentials for the estimator.

    Upper potentials follow the existing sign convention: solve the SE lower
    problem for -cost, then negate the resulting potentials.
    """
    t0 = time.time()
    lower = _solve_se_adjusted_lower_dual(
        probs0=probs0,
        probs1=probs1,
        cost=cost,
        pi=pi,
        n=n,
        solver=solver,
    )
    upper_neg = _solve_se_adjusted_lower_dual(
        probs0=probs0,
        probs1=probs1,
        cost=-np.asarray(cost, dtype=np.float64),
        pi=pi,
        n=n,
        solver=solver,
    )
    if diagnostics is not None:
        diagnostics.solve_count += 1
        diagnostics.runtime_seconds += time.time() - t0

    warning_count = lower.warning_count + upper_neg.warning_count
    status = "ok" if warning_count == 0 else "warning"
    return SEDualBoundPair(
        lower_value=lower.value,
        upper_value=-upper_neg.value,
        lower_u0=lower.u0,
        lower_u1=lower.u1,
        upper_u0=-upper_neg.u0,
        upper_u1=-upper_neg.u1,
        warning_count=warning_count,
        status=status,
    )


def compute_oracle_sharp_bounds(
    probs0: np.ndarray,
    probs1: np.ndarray,
    support_y: np.ndarray,
    cost: np.ndarray,
    max_units: int | None = None,
    num_threads: int | str = 1,
    num_iter_max: int = 100000,
    cache: dict[tuple[object, ...], OTBoundPair] | None = None,
    diagnostics: OTDiagnostics | None = None,
    progress_every: int = 0,
    quiet: bool = True,
    progress_label: str = "oracle benchmark",
) -> dict[str, float | str | int]:
    n = probs0.shape[0]
    if max_units is None or max_units >= n:
        eval_idx = np.arange(n)
        mode = "full"
    elif max_units <= 0:
        return {
            "oracle_lower_first_term": np.nan,
            "oracle_upper_first_term": np.nan,
            "true_ATE_norm_sq": np.nan,
            "oracle_lower_VarITE": np.nan,
            "oracle_upper_VarITE": np.nan,
            "oracle_computation_mode": "skipped",
            "oracle_n_units": 0,
            "oracle_solver_warning_count": 0,
            "oracle_ot_solve_count": 0,
            "oracle_ot_cache_hits": 0,
            "oracle_ot_runtime_seconds": 0.0,
            "oracle_runtime_seconds": 0.0,
        }
    else:
        eval_idx = np.arange(max_units)
        mode = "subsampled"

    t0 = time.time()
    local_diagnostics = diagnostics if diagnostics is not None else OTDiagnostics()
    lower_vals = []
    upper_vals = []
    warning_count = 0
    for row, i in enumerate(eval_idx):
        ot_pair = solve_ot_bound_pair(
            probs0[i],
            probs1[i],
            cost,
            num_threads=num_threads,
            num_iter_max=num_iter_max,
            cache=cache,
            diagnostics=local_diagnostics,
        )
        lower_vals.append(ot_pair.lower_value)
        upper_vals.append(ot_pair.upper_value)
        warning_count += ot_pair.warning_count
        _maybe_print_ot_progress(
            label=progress_label,
            row=row,
            total=len(eval_idx),
            progress_every=progress_every,
            quiet=quiet,
            diagnostics=local_diagnostics,
        )

    mu0 = expected_y(probs0[eval_idx], support_y)
    mu1 = expected_y(probs1[eval_idx], support_y)
    ate = (mu1 - mu0).mean(axis=0)
    ate_norm_sq = float(ate @ ate)
    lower_first = float(np.mean(lower_vals))
    upper_first = float(np.mean(upper_vals))

    return {
        "oracle_lower_first_term": lower_first,
        "oracle_upper_first_term": upper_first,
        "true_ATE_norm_sq": ate_norm_sq,
        "oracle_lower_VarITE": lower_first - ate_norm_sq,
        "oracle_upper_VarITE": upper_first - ate_norm_sq,
        "oracle_computation_mode": mode,
        "oracle_n_units": int(len(eval_idx)),
        "oracle_solver_warning_count": int(warning_count),
        "oracle_ot_solve_count": int(local_diagnostics.solve_count),
        "oracle_ot_cache_hits": int(local_diagnostics.cache_hits),
        "oracle_ot_runtime_seconds": float(local_diagnostics.runtime_seconds),
        "oracle_runtime_seconds": elapsed(t0),
    }


# ---------------------------------------------------------------------------
# Dual-bound estimator
# ---------------------------------------------------------------------------


def _split_indices(
    n: int,
    rng: np.random.Generator,
    crossfit: bool,
    n_folds: int,
    split_frac: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    perm = rng.permutation(n)
    if crossfit:
        folds = np.array_split(perm, n_folds)
        pairs = []
        for fold in folds:
            eval_idx = np.sort(fold)
            train_idx = np.sort(np.setdiff1d(np.arange(n), eval_idx, assume_unique=False))
            pairs.append((train_idx, eval_idx))
        return pairs

    split = int(np.floor(split_frac * n))
    train_idx = np.sort(perm[:split])
    eval_idx = np.sort(perm[split:])
    return [(train_idx, eval_idx)]


def _cap_eval_indices(
    n: int,
    split_pairs: list[tuple[np.ndarray, np.ndarray]],
    max_eval_units: int | None,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    all_eval = np.concatenate([eval_idx for _, eval_idx in split_pairs])
    if max_eval_units is None or max_eval_units >= len(all_eval):
        return split_pairs, "full", int(len(all_eval))

    keep = set(rng.choice(all_eval, size=max_eval_units, replace=False).tolist())
    capped = []
    for train_idx, eval_idx in split_pairs:
        mask = np.array([i in keep for i in eval_idx])
        capped.append((train_idx, eval_idx[mask]))
    return capped, "subsampled", int(max_eval_units)


def _first_term_summands_for_eval_units(
    data: BinaryVectorDGPData,
    eval_idx: np.ndarray,
    probs0_eval: np.ndarray,
    probs1_eval: np.ndarray,
    cost: np.ndarray,
    aipw: bool,
    num_threads: int | str,
    num_iter_max: int,
    cache: dict[tuple[object, ...], OTBoundPair] | None,
    diagnostics: OTDiagnostics | None,
    progress_every: int,
    quiet: bool,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    n_eval = len(eval_idx)
    lower_sbeta = np.zeros(n_eval)
    upper_sbeta = np.zeros(n_eval)
    mu0 = expected_y(probs0_eval, data.support_Y)
    mu1 = expected_y(probs1_eval, data.support_Y)
    warning_count = 0
    status = "ok"

    for row, i in enumerate(eval_idx):
        pi = data.propensity
        dual_pair = solve_se_bound_pair(
            probs0_eval[row],
            probs1_eval[row],
            cost,
            pi=pi,
            n=len(data.W),
            diagnostics=diagnostics,
        )
        warning_count += dual_pair.warning_count
        if dual_pair.status != "ok":
            status = "warning"
        _maybe_print_ot_progress(
            label=progress_label,
            row=row,
            total=n_eval,
            progress_every=progress_every,
            quiet=quiet,
            diagnostics=diagnostics,
        )

        atom = data.atom_obs[i]
        w = data.W[i]

        lower_c0 = float(probs0_eval[row] @ dual_pair.lower_u0)
        lower_c1 = float(probs1_eval[row] @ dual_pair.lower_u1)
        upper_c0 = float(probs0_eval[row] @ dual_pair.upper_u0)
        upper_c1 = float(probs1_eval[row] @ dual_pair.upper_u1)

        if w == 0:
            lower_obs = dual_pair.lower_u0[atom]
            upper_obs = dual_pair.upper_u0[atom]
            lower_sbeta[row] = lower_obs / (1 - pi)
            upper_sbeta[row] = upper_obs / (1 - pi)
            if aipw:
                lower_sbeta[row] = (lower_obs - lower_c0) / (1 - pi) + lower_c0 + lower_c1
                upper_sbeta[row] = (upper_obs - upper_c0) / (1 - pi) + upper_c0 + upper_c1
        else:
            lower_obs = dual_pair.lower_u1[atom]
            upper_obs = dual_pair.upper_u1[atom]
            lower_sbeta[row] = lower_obs / pi
            upper_sbeta[row] = upper_obs / pi
            if aipw:
                lower_sbeta[row] = (lower_obs - lower_c1) / pi + lower_c0 + lower_c1
                upper_sbeta[row] = (upper_obs - upper_c1) / pi + upper_c0 + upper_c1

    return lower_sbeta, upper_sbeta, mu0, mu1, warning_count, status


def estimate_varite_dual_bounds(
    data: BinaryVectorDGPData,
    cost: np.ndarray,
    model_type: str,
    crossfit: bool,
    n_folds: int,
    split_frac: float,
    aipw: bool,
    dual_max_units: int | None,
    smoothing_epsilon: float,
    logit_c: float,
    logit_solver: str,
    logit_max_iter: int,
    alpha: float,
    seed: int,
    num_threads: int | str,
    num_iter_max: int = 100000,
    cache: dict[tuple[object, ...], OTBoundPair] | None = None,
    diagnostics: OTDiagnostics | None = None,
    progress_every: int = 0,
    quiet: bool = True,
    progress_label: str = "dual estimator",
) -> dict[str, float | int | str | bool]:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    local_diagnostics = diagnostics if diagnostics is not None else OTDiagnostics()
    split_pairs = _split_indices(
        n=len(data.W),
        rng=rng,
        crossfit=crossfit,
        n_folds=n_folds,
        split_frac=split_frac,
    )
    split_pairs, dual_mode, n_eval = _cap_eval_indices(
        n=len(data.W),
        split_pairs=split_pairs,
        max_eval_units=dual_max_units,
        rng=rng,
    )

    lower_sbetas = []
    upper_sbetas = []
    skappa1s = []
    skappa0s = []
    solver_warning_count = 0
    model_warning_count = 0
    unique_arm0 = []
    unique_arm1 = []
    solver_status = "ok"

    for fold_id, (train_idx, eval_idx) in enumerate(split_pairs):
        if len(eval_idx) == 0:
            continue

        if model_type == "oracle":
            probs0_eval = data.probs0_true[eval_idx]
            probs1_eval = data.probs1_true[eval_idx]
            unique_arm0.append(int(len(np.unique(data.atom_obs[train_idx[data.W[train_idx] == 0]]))))
            unique_arm1.append(int(len(np.unique(data.atom_obs[train_idx[data.W[train_idx] == 1]]))))
        elif model_type == "estimated":
            probs0_eval, probs1_eval, model_diagnostics = fit_predict_two_arm_pmfs(
                data=data,
                train_idx=train_idx,
                eval_idx=eval_idx,
                smoothing_epsilon=smoothing_epsilon,
                logit_c=logit_c,
                logit_solver=logit_solver,
                logit_max_iter=logit_max_iter,
                seed=seed + 1009 * fold_id,
            )
            unique_arm0.append(model_diagnostics["observed_unique_atoms_arm0"])
            unique_arm1.append(model_diagnostics["observed_unique_atoms_arm1"])
            model_warning_count += model_diagnostics["model_warning_count"]
        else:
            raise ValueError(f"Unrecognized model_type={model_type}")

        lower_sb, upper_sb, mu0, mu1, warn_count, status = _first_term_summands_for_eval_units(
            data=data,
            eval_idx=eval_idx,
            probs0_eval=probs0_eval,
            probs1_eval=probs1_eval,
            cost=cost,
            aipw=aipw,
            num_threads=num_threads,
            num_iter_max=num_iter_max,
            cache=cache,
            diagnostics=local_diagnostics,
            progress_every=progress_every,
            quiet=quiet,
            progress_label=f"{progress_label} fold={fold_id + 1}/{len(split_pairs)}",
        )
        solver_warning_count += warn_count
        if status != "ok":
            solver_status = "warning"

        pi = data.propensity
        y_eval = data.Y[eval_idx].astype(np.float64)
        w_eval = data.W[eval_idx].reshape(-1, 1)
        if aipw:
            kappa1 = w_eval * (y_eval - mu1) / pi + mu1
            kappa0 = (1 - w_eval) * (y_eval - mu0) / (1 - pi) + mu0
        else:
            kappa1 = w_eval * y_eval / pi
            kappa0 = (1 - w_eval) * y_eval / (1 - pi)

        lower_sbetas.append(lower_sb)
        upper_sbetas.append(upper_sb)
        skappa1s.append(kappa1)
        skappa0s.append(kappa0)

    lower_sbetas = np.concatenate(lower_sbetas)
    upper_sbetas = np.concatenate(upper_sbetas)
    skappa1s = np.vstack(skappa1s)
    skappa0s = np.vstack(skappa0s)

    lower_varite, lower_se = vector_varite_delta_method_se(lower_sbetas, skappa1s, skappa0s)
    upper_varite, upper_se = vector_varite_delta_method_se(upper_sbetas, skappa1s, skappa0s)
    ate_vec = skappa1s.mean(axis=0) - skappa0s.mean(axis=0)
    ate_norm_sq_est = float(ate_vec @ ate_vec)
    scale = stats.norm.ppf(1 - alpha / 2)

    return {
        "model_type": model_type,
        "crossfit": bool(crossfit),
        "n_folds": int(n_folds if crossfit else 1),
        "split_frac": float(split_frac),
        "aipw": bool(aipw),
        "n_eval": int(len(lower_sbetas)),
        "dual_computation_mode": dual_mode,
        "runtime_seconds": elapsed(t0),
        "ot_solve_count": int(local_diagnostics.solve_count),
        "ot_cache_hits": int(local_diagnostics.cache_hits),
        "ot_runtime_seconds": float(local_diagnostics.runtime_seconds),
        "solver_status": solver_status,
        "solver_warning_count": int(solver_warning_count),
        "model_warning_count": int(model_warning_count),
        "observed_unique_atoms_arm0": float(np.mean(unique_arm0)) if unique_arm0 else np.nan,
        "observed_unique_atoms_arm1": float(np.mean(unique_arm1)) if unique_arm1 else np.nan,
        "dual_lower_first_term_estimate": float(lower_sbetas.mean()),
        "dual_upper_first_term_estimate": float(upper_sbetas.mean()),
        "dual_lower_first_term_se": float(lower_sbetas.std(ddof=1) / np.sqrt(len(lower_sbetas))),
        "dual_upper_first_term_se": float(upper_sbetas.std(ddof=1) / np.sqrt(len(upper_sbetas))),
        "dual_ATE_norm_sq_estimate": ate_norm_sq_est,
        "dual_lower_VarITE_estimate": float(lower_varite),
        "dual_upper_VarITE_estimate": float(upper_varite),
        "dual_lower_se": float(lower_se),
        "dual_upper_se": float(upper_se),
        "dual_lower_ci": float(lower_varite - scale * lower_se),
        "dual_upper_ci": float(upper_varite + scale * upper_se),
    }


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def reps_for_n(n: int, args) -> int:
    if args.reps is not None:
        return int(args.reps)
    if n <= 100:
        return int(args.reps_small)
    if n <= 500:
        return int(args.reps_medium)
    return int(args.reps_large)


def model_types_from_arg(model_type: str) -> list[str]:
    if model_type == "both":
        return ["oracle", "estimated"]
    return [model_type]


def run_single_repetition(n: int, rep_id: int, args, cost: np.ndarray) -> list[dict]:
    seed_rep = int(args.master_seed + 100000 * rep_id + n)
    data = generate_binary_vector_dgp(
        n=n,
        p_x=args.p_x,
        dim_y=args.dim_y,
        propensity=args.propensity,
        alpha_scale=args.alpha_scale,
        beta_scale=args.beta_scale,
        seed=seed_rep,
    )
    ot_cache: dict[tuple[object, ...], OTBoundPair] = {}

    if args.skip_oracle:
        oracle = {
            "oracle_lower_first_term": np.nan,
            "oracle_upper_first_term": np.nan,
            "true_ATE_norm_sq": np.nan,
            "oracle_lower_VarITE": np.nan,
            "oracle_upper_VarITE": np.nan,
            "oracle_computation_mode": "skipped",
            "oracle_n_units": 0,
            "oracle_solver_warning_count": 0,
            "oracle_ot_solve_count": 0,
            "oracle_ot_cache_hits": 0,
            "oracle_ot_runtime_seconds": 0.0,
            "oracle_runtime_seconds": 0.0,
        }
    else:
        oracle_diagnostics = OTDiagnostics()
        oracle = compute_oracle_sharp_bounds(
            probs0=data.probs0_true,
            probs1=data.probs1_true,
            support_y=data.support_Y,
            cost=cost,
            max_units=args.oracle_max_units,
            num_threads=args.ot_num_threads,
            num_iter_max=args.ot_max_iter,
            cache=ot_cache,
            diagnostics=oracle_diagnostics,
            progress_every=args.progress_every,
            quiet=args.quiet,
            progress_label=f"n={n} rep={rep_id} oracle benchmark",
        )

    rows = []
    for model_type in model_types_from_arg(args.model_type):
        estimate_diagnostics = OTDiagnostics()
        estimates = estimate_varite_dual_bounds(
            data=data,
            cost=cost,
            model_type=model_type,
            crossfit=args.crossfit,
            n_folds=args.n_folds,
            split_frac=args.split_frac,
            aipw=args.aipw,
            dual_max_units=args.dual_max_units,
            smoothing_epsilon=args.smoothing_epsilon,
            logit_c=args.logit_c,
            logit_solver=args.logit_solver,
            logit_max_iter=args.logit_max_iter,
            alpha=args.alpha,
            seed=seed_rep + 911,
            num_threads=args.ot_num_threads,
            num_iter_max=args.ot_max_iter,
            cache=ot_cache,
            diagnostics=estimate_diagnostics,
            progress_every=args.progress_every,
            quiet=args.quiet,
            progress_label=f"n={n} rep={rep_id} model={model_type}",
        )
        row = {
            "n": int(n),
            "rep_id": int(rep_id),
            "seed": seed_rep,
            "dim_y": int(args.dim_y),
            "support_size": int(data.support_Y.shape[0]),
            "p_x": int(args.p_x),
            "propensity": float(args.propensity),
            "beta_scale": float(args.beta_scale),
            "alpha_scale": float(args.alpha_scale),
            "smoothing_epsilon": float(args.smoothing_epsilon),
            **estimates,
            **oracle,
        }
        can_check_coverage = (
            np.isfinite(row["oracle_lower_VarITE"])
            and row["oracle_computation_mode"] == "full"
            and row["dual_computation_mode"] == "full"
        )
        row["dual_lower_covers_oracle"] = (
            bool(row["dual_lower_ci"] <= row["oracle_lower_VarITE"])
            if can_check_coverage
            else np.nan
        )
        row["dual_upper_covers_oracle"] = (
            bool(row["dual_upper_ci"] >= row["oracle_upper_VarITE"])
            if can_check_coverage
            else np.nan
        )
        row["any_anticonservative"] = (
            bool((not row["dual_lower_covers_oracle"]) or (not row["dual_upper_covers_oracle"]))
            if can_check_coverage
            else np.nan
        )
        rows.append(row)
    return rows


def write_rows_csv(rows: list[dict], output_csv: str | None) -> None:
    if not output_csv:
        return
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_row_summary(row: dict) -> None:
    msg = (
        f"n={row['n']} rep={row['rep_id']} model={row['model_type']} "
        f"eval={row['n_eval']} mode={row['dual_computation_mode']} "
        f"lower={row['dual_lower_VarITE_estimate']:.4f} "
        f"upper={row['dual_upper_VarITE_estimate']:.4f} "
        f"ci=({row['dual_lower_ci']:.4f}, {row['dual_upper_ci']:.4f}) "
        f"runtime={row['runtime_seconds']:.3f}s "
        f"solver={row['ot_runtime_seconds']:.3f}s "
        f"solves={row['ot_solve_count']} "
        f"cache_hits={row['ot_cache_hits']}"
    )
    if np.isfinite(row["oracle_lower_VarITE"]):
        msg += (
            f" oracle=({row['oracle_lower_VarITE']:.4f}, "
            f"{row['oracle_upper_VarITE']:.4f}) "
            f"oracle_rt={row['oracle_runtime_seconds']:.3f}s"
        )
    print(msg)


def run_experiment(args) -> list[dict]:
    support_y = make_binary_support(args.dim_y)
    cost = build_vector_cost_matrix(support_y, support_y)
    all_rows = []
    for n in args.n_list:
        reps = reps_for_n(n, args)
        for rep_id in range(reps):
            if not args.quiet:
                print(f"Starting n={n} rep={rep_id}/{reps - 1}", flush=True)
            rows = run_single_repetition(n=n, rep_id=rep_id, args=args, cost=cost)
            all_rows.extend(rows)
            if args.output_csv and args.checkpoint_every_rep:
                write_rows_csv(all_rows, args.output_csv)
            if not args.quiet:
                for row in rows:
                    print_row_summary(row)
    write_rows_csv(all_rows, args.output_csv)
    if args.output_csv and not args.quiet:
        print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    return all_rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class DiscreteVectorVarITETests(unittest.TestCase):
    def test_make_binary_support(self):
        support = make_binary_support(10)
        self.assertEqual(support.shape, (1024, 10))
        self.assertTrue(np.all((support == 0) | (support == 1)))
        self.assertEqual(np.unique(support, axis=0).shape[0], 1024)

    def test_build_vector_cost_matrix(self):
        y0 = np.array([[0, 0, 1], [1, 0, 1]])
        y1 = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
        cost = build_vector_cost_matrix(y0, y1)
        self.assertEqual(cost.shape, (2, 3))
        self.assertEqual(cost[0, 0], 1)
        self.assertEqual(cost[0, 1], 3)
        self.assertEqual(cost[1, 2], 1)

    def test_dgp_shapes_and_probabilities(self):
        data = generate_binary_vector_dgp(n=20, p_x=5, dim_y=4, seed=123)
        self.assertEqual(data.support_Y.shape, (16, 4))
        self.assertEqual(data.Y.shape, (20, 4))
        self.assertEqual(data.Y0.shape, (20, 4))
        self.assertEqual(data.Y1.shape, (20, 4))
        self.assertEqual(data.atom0.shape, (20,))
        self.assertEqual(data.atom1.shape, (20,))
        self.assertEqual(data.atom_obs.shape, (20,))
        self.assertEqual(data.probs0_true.shape, (20, 16))
        self.assertEqual(data.probs1_true.shape, (20, 16))
        np.testing.assert_allclose(data.probs0_true.sum(axis=1), 1)
        np.testing.assert_allclose(data.probs1_true.sum(axis=1), 1)

    def test_multinomial_missing_classes_smoothing(self):
        rng = np.random.default_rng(1)
        X_train = rng.normal(size=(8, 3))
        atom_train = np.array([0, 0, 2, 2, 2, 5, 5, 5])
        X_eval = rng.normal(size=(4, 3))
        pred = fit_predict_multinomial_pmf(
            X_train=X_train,
            atom_train=atom_train,
            X_eval=X_eval,
            support_size=16,
            smoothing_epsilon=1e-6,
            seed=1,
        )
        self.assertEqual(pred.probs.shape, (4, 16))
        self.assertEqual(pred.observed_unique_atoms, 3)
        self.assertTrue(np.all(pred.probs > 0))
        np.testing.assert_allclose(pred.probs.sum(axis=1), 1)

    def test_small_oracle_estimator_runs(self):
        data = generate_binary_vector_dgp(n=24, p_x=4, dim_y=3, seed=12)
        cost = build_vector_cost_matrix(data.support_Y, data.support_Y)
        result = estimate_varite_dual_bounds(
            data=data,
            cost=cost,
            model_type="oracle",
            crossfit=True,
            n_folds=2,
            split_frac=0.5,
            aipw=False,
            dual_max_units=10,
            smoothing_epsilon=1e-6,
            logit_c=1.0,
            logit_solver="lbfgs",
            logit_max_iter=100,
            alpha=0.05,
            seed=1,
            num_threads=1,
        )
        self.assertEqual(result["n_eval"], 10)
        self.assertTrue(np.isfinite(result["dual_lower_VarITE_estimate"]))
        self.assertTrue(np.isfinite(result["dual_upper_VarITE_estimate"]))
        self.assertTrue(np.isfinite(result["dual_lower_ci"]))
        self.assertTrue(np.isfinite(result["dual_upper_ci"]))
        self.assertLessEqual(
            result["dual_lower_first_term_estimate"],
            result["dual_upper_first_term_estimate"] + 1e-8,
        )

    def test_se_dual_potentials_run_on_small_support(self):
        rng = np.random.default_rng(123)
        diagnostics = OTDiagnostics()
        for dim_y in (2, 3):
            support = make_binary_support(dim_y)
            cost = build_vector_cost_matrix(support, support)
            probs0 = _normalize_prob_vector(rng.random(support.shape[0]))
            probs1 = _normalize_prob_vector(rng.random(support.shape[0]))

            pair = solve_se_bound_pair(
                probs0=probs0,
                probs1=probs1,
                cost=cost,
                pi=0.3,
                n=30,
                diagnostics=diagnostics,
            )
            lower_first = float(probs0 @ pair.lower_u0 + probs1 @ pair.lower_u1)
            upper_first = float(probs0 @ pair.upper_u0 + probs1 @ pair.upper_u1)

            self.assertEqual(pair.status, "ok")
            self.assertTrue(np.isfinite(pair.lower_value))
            self.assertTrue(np.isfinite(pair.upper_value))
            self.assertTrue(np.all(np.isfinite(pair.lower_u0)))
            self.assertTrue(np.all(np.isfinite(pair.lower_u1)))
            self.assertTrue(np.all(np.isfinite(pair.upper_u0)))
            self.assertTrue(np.all(np.isfinite(pair.upper_u1)))
            self.assertLessEqual(lower_first, upper_first + 1e-7)

    def test_ot_cache_reuses_solution(self):
        data = generate_binary_vector_dgp(n=8, p_x=3, dim_y=3, seed=22)
        cost = build_vector_cost_matrix(data.support_Y, data.support_Y)
        cache: dict[tuple[object, ...], OTBoundPair] = {}

        first_diagnostics = OTDiagnostics()
        first = solve_ot_bound_pair(
            data.probs0_true[0],
            data.probs1_true[0],
            cost,
            cache=cache,
            diagnostics=first_diagnostics,
        )
        second_diagnostics = OTDiagnostics()
        second = solve_ot_bound_pair(
            data.probs0_true[0],
            data.probs1_true[0],
            cost,
            cache=cache,
            diagnostics=second_diagnostics,
        )

        self.assertEqual(first_diagnostics.solve_count, 1)
        self.assertEqual(first_diagnostics.cache_hits, 0)
        self.assertEqual(second_diagnostics.solve_count, 0)
        self.assertEqual(second_diagnostics.cache_hits, 1)
        self.assertEqual(first.lower_value, second.lower_value)


def run_tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiscreteVectorVarITETests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-dimensional discrete vector VarITE dual-bounds experiment."
    )
    parser.add_argument("command", nargs="?", default="run", choices=["run", "test"])

    parser.add_argument("--n-list", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--reps-small", type=int, default=50)
    parser.add_argument("--reps-medium", type=int, default=10)
    parser.add_argument("--reps-large", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=1)

    parser.add_argument("--dim-y", type=int, default=10)
    parser.add_argument("--p-x", type=int, default=50)
    parser.add_argument("--propensity", type=float, default=0.3)
    parser.add_argument("--alpha-scale", type=float, default=0.5)
    parser.add_argument("--beta-scale", type=float, default=0.3)

    parser.add_argument("--model-type", choices=["oracle", "estimated", "both"], default="both")
    parser.add_argument("--crossfit", type=str2bool, default=True)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--split-frac", type=float, default=0.5)
    parser.add_argument("--aipw", type=str2bool, default=False)

    parser.add_argument("--smoothing-epsilon", type=float, default=1e-6)
    parser.add_argument("--logit-c", type=float, default=1.0)
    parser.add_argument("--logit-solver", choices=["lbfgs", "saga"], default="lbfgs")
    parser.add_argument("--logit-max-iter", type=int, default=200)

    parser.add_argument("--oracle-max-units", type=int, default=None)
    parser.add_argument("--skip-oracle", type=str2bool, default=False)
    parser.add_argument("--dual-max-units", type=int, default=None)
    parser.add_argument("--ot-num-threads", type=parse_ot_threads, default=1)
    parser.add_argument("--ot-max-iter", type=int, default=100000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--output-csv", default="varite_discrete_experiment_results.csv")
    parser.add_argument("--checkpoint-every-rep", type=str2bool, default=True)
    parser.add_argument("--quiet", type=str2bool, default=False)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.ot_max_iter <= 0:
        parser.error("--ot-max-iter must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.command == "test":
        run_tests()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
