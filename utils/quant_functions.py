import numpy as np
import pandas as pd

# ===========================
# BASIC RETURN CALCULATIONS
# ===========================

def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Computes simple returns: (P_t - P_(t-1)) / P_(t-1)
    """
    return prices.pct_change().dropna()

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Computes log returns: ln(P_t / P_(t-1))
    """
    return np.log(prices / prices.shift(1)).dropna()

# ===========================
# PORTFOLIO STATISTICS
# ===========================

def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Computes annualized portfolio return.
    
    weights: array of portfolio weights
    returns: dataframe of asset returns (daily)
    """
    mean_daily = returns.mean()
    return np.dot(weights, mean_daily) * 252  # annualized


def portfolio_variance(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Computes annualized portfolio variance.
    
    weights: array of portfolio weights
    returns: dataframe of asset returns (daily)
    """
    cov_matrix = returns.cov() * 252  # annualized covariance
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def portfolio_std(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Computes annualized portfolio standard deviation.
    """
    return np.sqrt(portfolio_variance(weights, returns))


def sharpe_ratio(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """
    Computes Sharpe ratio: (rp - rf) / sigma_p
    rf default: 2% annual
    """
    rp = portfolio_return(weights, returns)
    sigma = portfolio_std(weights, returns)
    return (rp - risk_free_rate) / sigma


# ===========================
# EFFICIENT FRONTIER HELPERS
# ===========================

def generate_random_weights(n_assets: int) -> np.ndarray:
    """
    Generates random portfolio weights that sum to 1.
    Useful for Monte Carlo sampling of portfolios.
    """
    w = np.random.random(n_assets)
    return w / w.sum()


def random_portfolio_performance(returns: pd.DataFrame, n_samples: int = 5000):
    """
    Generates random portfolios and returns:
    - list of returns
    - list of volatilities
    - list of weights
    """
    n_assets = returns.shape[1]

    rets = []
    vols = []
    wts = []

    for _ in range(n_samples):
        w = generate_random_weights(n_assets)
        r = portfolio_return(w, returns)
        v = portfolio_std(w, returns)

        rets.append(r)
        vols.append(v)
        wts.append(w)

    return np.array(rets), np.array(vols), wts

import cvxpy as cp

# ===========================
# PORTFOLIO OPTIMIZATION
# ===========================

def global_min_variance(returns: pd.DataFrame) -> np.ndarray:
    """
    Computes the Global Minimum Variance (GMV) portfolio.
    Minimizes portfolio volatility with the constraint sum(weights)=1.
    """
    n = returns.shape[1]
    cov = returns.cov().values * 252  # annualized covariance

    w = cp.Variable(n)
    objective = cp.quad_form(w, cov)  # variance
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    return w.value


def max_sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.02) -> np.ndarray:
    """
    Computes the maximum Sharpe ratio portfolio in a DCP-compliant way.
    Method:
    - Sweep target returns
    - Solve a convex optimization for each (min variance)
    - Compute Sharpe for each solution
    - Pick the best one
    """
    n = returns.shape[1]
    mean_ret = returns.mean().values * 252
    cov = returns.cov().values * 252

    target_grid = np.linspace(mean_ret.min(), mean_ret.max(), 50)

    best_sharpe = -1
    best_weights = None

    for target in target_grid:
        w = cp.Variable(n)
        variance = cp.quad_form(w, cov)
        constraints = [
            cp.sum(w) == 1,
            mean_ret @ w == target,
            w >= 0
        ]
        prob = cp.Problem(cp.Minimize(variance), constraints)

        try:
            prob.solve()
            if w.value is None:
                continue

            vol = np.sqrt(prob.value)
            ret = target
            sharpe = (ret - risk_free_rate) / vol

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w.value

        except:
            continue

    return best_weights



def efficient_frontier(returns, num_points=80, target_returns=None):
    """
    Clean, stable efficient frontier:
    Always smooth, monotonic, convex.
    Returns [{"x": vol%, "y": ret%}]
    """

    mu = returns.mean().values * 252      # expected annual returns (fraction)
    cov = returns.cov().values * 252      # annual covariance
    n = len(mu)

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)
    base = [cp.sum(w) == 1, w >= 0]

    tr_min = float(mu.min())
    tr_max = float(mu.max())
    eps = 1e-6
    if target_returns is None:
        target_grid = np.linspace(tr_min + eps, tr_max - eps, num_points)
    else:
        target_grid = np.asarray(list(target_returns), float)

    frontier = []

    for tr in target_grid:

    # tolerance band to avoid infeasible equality constraints
            # tolerance band to avoid infeasible equality constraints
        tol = 1e-4

        constraints = base + [
            ret >= tr - tol,
            ret <= tr + tol
        ]

        prob = cp.Problem(cp.Minimize(risk), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            continue

        if w.value is None:
            continue

        weights = np.asarray(w.value).flatten()
        if not np.all(np.isfinite(weights)):
            continue

        port_ret = float(mu @ weights) * 100.0
        port_vol = float(np.sqrt(weights.T @ cov @ weights)) * 100.0

        frontier.append({"x": port_vol, "y": port_ret})


    

    return frontier

import matplotlib.pyplot as plt

# ===========================
# PLOTTING UTILITIES
# ===========================

def plot_random_portfolios(returns, n_samples=5000):
    """
    Plots random portfolios based on Monte Carlo sampling.
    """
    rets, vols, _ = random_portfolio_performance(returns, n_samples)

    plt.figure(figsize=(10, 6))
    plt.scatter(vols, rets, alpha=0.3, s=10)
    plt.xlabel("Volatility (Std Dev)")
    plt.ylabel("Return")
    plt.title("Random Portfolios")
    plt.grid(True)


def plot_efficient_frontier(returns, save_path=None):
    """
    Plots the efficient frontier along with GMV and Max Sharpe portfolios.
    This function was updated to work with efficient_frontier() which now returns
    a list of Chart.js-friendly points in percent units: [{"x":vol%, "y":ret%}, ...].
    """
    # Random portfolios for background
    rets, vols, _ = random_portfolio_performance(returns, 3000)

    # convert to percent units for plotting consistency
    rets_pct = np.array(rets) * 100.0
    vols_pct = np.array(vols) * 100.0

    # Optimization results
    gmv = global_min_variance(returns)
    msr = max_sharpe_ratio(returns)

    gmv_ret = portfolio_return(gmv, returns) * 100.0
    gmv_vol = portfolio_std(gmv, returns) * 100.0

    msr_ret = portfolio_return(msr, returns) * 100.0
    msr_vol = portfolio_std(msr, returns) * 100.0

    # Frontier targets (annual returns as decimals)
    # Correct, stable target-return grid
    mu = returns.mean().values * 252  # annual expected returns

    target_returns = np.linspace(mu.min(), mu.max(), 80)

    # Correct call using keyword argument
    ef_points = efficient_frontier(returns, target_returns=target_returns)


    # extract vols and returns from ef_points (they are in percent)
    ef_vols = [p['x'] for p in ef_points]
    ef_returns = [p['y'] for p in ef_points]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(vols_pct, rets_pct, alpha=0.2, s=10, label="Random Portfolios")
    if ef_vols and ef_returns:
        plt.plot(ef_vols, ef_returns, color="red", linewidth=2, label="Efficient Frontier")
    plt.scatter(gmv_vol, gmv_ret, color="green", s=80, label="GMV Portfolio")
    plt.scatter(msr_vol, msr_ret, color="blue", s=80, label="Max Sharpe Portfolio")

    plt.xlabel("Volatility (%)")
    plt.ylabel("Return (%)")
    plt.title("Efficient Frontier with GMV & Max Sharpe Portfolios")

    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
