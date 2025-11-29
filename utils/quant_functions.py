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
