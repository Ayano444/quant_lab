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


