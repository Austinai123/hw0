"""Mean-variance frontiers, with and without short sales.

Notation: `mu` is the vector of expected returns, `Sigma` is the covariance
matrix of returns, and `w` is a vector of portfolio weights that sums to one.

The frontier problem is

    minimize    w' Sigma w
    subject to  w' mu = target,   sum(w) = 1.

With only these two equality constraints, the problem has a closed-form
solution (`efficient_weights`). Add the no-short-sales constraint `w >= 0` and
it no longer does, so we solve it numerically (`minimize_variance`). The
notebooks `01_markowitz.ipynb.py` and `02_markowitz_derivation.ipynb.py`
explain why.

Add a risk-free asset and the frontier becomes a pair of straight lines, again
with a closed form (`risk_free_frontier`). Rule out short sales or borrowing
and it is back to a numerical solution (`minimize_variance_with_risk_free`).
"""

import numpy as np
from scipy.optimize import minimize


def frontier_constants(mu, Sigma):
    """The scalars A, B, C, D that summarize the unconstrained frontier."""
    ones = np.ones(len(mu))
    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    Sigma_inv_mu = np.linalg.solve(Sigma, mu)
    A = ones @ Sigma_inv_ones
    B = ones @ Sigma_inv_mu
    C = mu @ Sigma_inv_mu
    D = A * C - B**2
    return A, B, C, D


def efficient_weights(mu, Sigma, target):
    """Closed-form minimum variance weights for a target mean (shorts allowed)."""
    ones = np.ones(len(mu))
    A, B, C, D = frontier_constants(mu, Sigma)
    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    Sigma_inv_mu = np.linalg.solve(Sigma, mu)
    return ((C - B * target) * Sigma_inv_ones + (A * target - B) * Sigma_inv_mu) / D


def frontier_variance(mu, Sigma, targets):
    """Closed-form minimum variance for each target mean (shorts allowed)."""
    A, B, C, D = frontier_constants(mu, Sigma)
    targets = np.asarray(targets)
    return (A * targets**2 - 2 * B * targets + C) / D


def minimize_variance(mu, Sigma, target=None, allow_short=True):
    """Numerically solve for the minimum variance weights.

    If `target` is None, there is no constraint on the mean, which gives the
    global minimum variance portfolio. If `allow_short` is False, every weight
    is constrained to be nonnegative.
    """
    n = len(mu)
    scale = 1 / np.mean(np.diag(Sigma))  # keeps the objective near 1 for the solver

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.ones(n) / n
    if target is not None:
        spread = mu.max() - mu.min()
        constraints.append({"type": "eq", "fun": lambda w: (w @ mu - target) / spread})
        # Start from a portfolio that already hits the target: a mix of the
        # lowest-mean and highest-mean assets.
        share = (target - mu.min()) / spread
        x0 = np.zeros(n)
        x0[mu.argmin()] += 1 - share
        x0[mu.argmax()] += share

    result = minimize(
        fun=lambda w: scale * w @ Sigma @ w,
        jac=lambda w: 2 * scale * Sigma @ w,
        x0=x0,
        bounds=None if allow_short else [(0, None)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = result.x
    # At the ends of the no-short frontier only one portfolio is feasible, and
    # the solver reports that it cannot improve. Accept any feasible answer.
    feasible = np.isclose(w.sum(), 1, atol=1e-8) and (
        target is None or np.isclose(w @ mu, target, atol=1e-8)
    )
    if not (result.success or feasible):
        raise RuntimeError(f"Optimizer failed: {result.message}")
    return w


def max_sharpe_weights(mu, Sigma, rf, allow_short=True):
    """Numerically solve for the portfolio with the highest Sharpe ratio.

    Uses a standard change of variables that turns the ratio into a quadratic
    program: minimize y' Sigma y subject to y' (mu - rf) = 1, then rescale y so
    that it sums to one.
    """
    n = len(mu)
    excess = mu - rf
    scale = 1 / np.mean(np.diag(Sigma))

    # Start from the single asset whose excess return is largest. Without
    # shorts it has to be an asset that beats the risk-free rate.
    best = np.abs(excess).argmax() if allow_short else excess.argmax()
    if excess[best] == 0 or (not allow_short and excess[best] < 0):
        raise RuntimeError("No asset has an expected return above the risk-free rate.")
    x0 = np.zeros(n)
    x0[best] = 1 / excess[best]

    result = minimize(
        fun=lambda y: scale * y @ Sigma @ y,
        jac=lambda y: 2 * scale * Sigma @ y,
        x0=x0,
        bounds=None if allow_short else [(0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda y: y @ excess - 1}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    y = result.x
    # As in `minimize_variance`, the solver can report failure after it has
    # reached the answer, and whether it does depends on the machine. What
    # matters is feasibility.
    feasible = np.isclose(y @ excess, 1, atol=1e-8) and (
        allow_short or y.min() >= -1e-8
    )
    if not (result.success or feasible):
        raise RuntimeError(f"Optimizer failed: {result.message}")
    return y / y.sum()


def risk_free_frontier(mu, Sigma, rf, tol=1e-10):
    """Closed-form frontier once there is a risk-free asset (shorts and borrowing allowed).

    Every frontier portfolio holds the risky assets in proportion to
    z = Sigma^{-1} (mu - rf) and puts the rest of its wealth in the risk-free
    asset. The frontier is the pair of lines mean = rf +/- sqrt(H) * vol, where
    H = (mu - rf)' z is the square of the highest attainable Sharpe ratio. The
    portfolio that targets a mean of m holds (m - rf) / H * z in the risky assets.

    Returns z, H, and whether there is an arbitrage. That can only happen when
    Sigma is singular: some combination of the risky assets then has zero
    variance, and unless it earns exactly the risk-free rate, any mean can be
    reached with no risk at all. Combinations with zero variance are left out
    of z.
    """
    excess = mu - rf
    eigval, eigvec = np.linalg.eigh(Sigma)
    risky = eigval > tol * eigval.max()
    riskless_excess = eigvec[:, ~risky].T @ excess
    arbitrage = bool(np.any(np.abs(riskless_excess) > 1e-8 * np.linalg.norm(excess)))
    z = eigvec[:, risky] @ (eigvec[:, risky].T @ excess / eigval[risky])
    H = excess @ z
    return z, H, arbitrage


def minimize_variance_with_risk_free(
    mu, Sigma, rf, target, allow_short=True, allow_borrowing=True
):
    """Numerically solve for the minimum variance risky weights given a risk-free asset.

    Whatever is not in the risky assets, 1 - w.sum(), is in the risk-free
    asset, so the risky weights no longer have to sum to one. If `allow_short`
    is False, no risky weight can be negative. If `allow_borrowing` is False,
    the weight on the risk-free asset cannot be negative. Raises a RuntimeError
    if the constraints rule out the target.
    """
    n = len(mu)
    excess = mu - rf
    gain = target - rf
    if gain == 0:
        return np.zeros(n)  # everything in the risk-free asset: no variance at all
    spread = np.abs(excess).max()
    if spread == 0:
        raise RuntimeError("Every asset earns the risk-free rate, so no other mean is feasible.")
    scale = 1 / np.mean(np.diag(Sigma))

    if allow_short:
        z, H, _ = risk_free_frontier(mu, Sigma, rf)
        # The closed form is the answer unless the borrowing constraint binds.
        x0 = gain / H * z if H > 0 else np.zeros(n)
    else:
        # Without shorts, the mean can only be pulled away from rf by assets on
        # the same side of rf as the target. Without borrowing as well, it can
        # be pulled no further than the most extreme of those assets.
        best = (np.sign(gain) * excess).argmax()
        reach = np.sign(gain) * excess[best]
        if reach <= 0 or (not allow_borrowing and abs(gain) > reach * (1 + 1e-12)):
            raise RuntimeError("The constraints rule out this target.")
        x0 = np.zeros(n)
        x0[best] = gain / excess[best]

    constraints = [{"type": "eq", "fun": lambda w: (w @ excess - gain) / spread}]
    if not allow_borrowing:
        constraints.append({"type": "ineq", "fun": lambda w: 1 - w.sum()})

    result = minimize(
        fun=lambda w: scale * w @ Sigma @ w,
        jac=lambda w: 2 * scale * Sigma @ w,
        x0=x0,
        bounds=None if allow_short else [(0, None)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = result.x
    # As in `minimize_variance`, the solver can report failure at a corner
    # where only one portfolio is feasible. What matters is feasibility.
    feasible = (
        np.isclose(w @ excess, gain, atol=1e-8)
        and (allow_short or w.min() >= -1e-8)
        and (allow_borrowing or w.sum() <= 1 + 1e-8)
    )
    if not feasible:
        raise RuntimeError(f"Optimizer failed: {result.message}")
    return w


def frontier(mu, Sigma, targets, allow_short=True):
    """Minimum variance weights for each target mean, one row per target."""
    if allow_short:
        return np.array([efficient_weights(mu, Sigma, m) for m in targets])
    return np.array(
        [minimize_variance(mu, Sigma, m, allow_short=False) for m in targets]
    )
