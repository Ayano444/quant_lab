import json
from django.shortcuts import render, redirect
from django.urls import reverse
import yfinance as yf
import pandas as pd
import numpy as np
import os, sys
from scipy.optimize import minimize
import cvxpy as cp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils.quant_functions import (
    global_min_variance,
    max_sharpe_ratio,
    efficient_frontier
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.conf import settings


# ----------------------------
# HOME PAGE
# ----------------------------
def home(request):
    return render(request, "home.html")


# ----------------------------
# OPTIMIZER SUITE INDEX PAGE
# ----------------------------
def optimizer_index(request):
    return render(request, "optimizer/index.html")


# ----------------------------
# CLASSIC OPTIMIZER INPUT PAGE
# ----------------------------
def classic_optimizer(request):
    # Set default values for the form
    context = {
        'tickers': request.GET.get('tickers', ''),
        'start_date': request.GET.get('start_date', ''),
        'end_date': request.GET.get('end_date', ''),
        'risk_free': request.GET.get('risk_free', '4.5'),
        'optimization_method': request.GET.get('optimization_method', 'min_volatility'),
        'error': request.GET.get('error', '')
    }
    return render(request, "optimizer/classic_input.html", context)


# ----------------------------
# CLASSIC OPTIMIZER PROCESSING
# ----------------------------
def classic_optimizer_run(request):
    # If user enters URL directly → redirect back to input page
    if request.method != "POST":
        return redirect("classic_optimizer")

    # Get form data with proper validation
    tickers = request.POST.get("tickers", "").strip()
    if not tickers:
        return redirect(f"{reverse('classic_optimizer')}?error=Please+enter+at+least+one+ticker+symbol")
        
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    start = request.POST.get("start_date")
    end = request.POST.get("end_date")
    
    # Set default values if not provided
    risk_free = request.POST.get("risk_free", "4.5")  # Default to 4.5% if not provided
    try:
        rf = float(risk_free) / 100
    except (ValueError, TypeError):
        rf = 0.045  # Default to 4.5% if invalid value
    
    # Get optimization method, default to min_volatility if not provided
    optimization_method = request.POST.get("optimization_method", "min_volatility")

    # Download Data with error handling
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)
        
        if data.empty:
            return redirect(
                f"{reverse('classic_optimizer')}?"
                f"error=No+data+returned.+Try+different+tickers+or+dates.&"
                f"tickers={','.join(tickers)}&"
                f"start_date={start}&"
                f"end_date={end}&"
                f"risk_free={risk_free}&"
                f"optimization_method={optimization_method}"
            )
    except Exception as e:
        return redirect(
            f"{reverse('classic_optimizer')}?"
            f"error=Error+downloading+data%3A+{str(e).replace(' ', '+')}&"
            f"tickers={','.join(tickers)}&"
            f"start_date={start}&"
            f"end_date={end}&"
            f"risk_free={risk_free}&"
            f"optimization_method={optimization_method}"
        )

    # Select prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]

    returns = prices.pct_change().dropna()

    # Optimization based on selected method
    if optimization_method == "max_sharpe":
        weights = max_sharpe_ratio(returns, risk_free_rate=rf)
        gmv = global_min_variance(returns)  # Still calculate GMV for comparison
        msr = weights  # Use the optimized weights for MSR
    else:  # Default to minimum volatility
        gmv = global_min_variance(returns)
        msr = max_sharpe_ratio(returns, risk_free_rate=rf)  # Still calculate MSR for comparison

    # Calculate daily statistics
    def portfolio_daily_stats(weights, is_msr=False):
        port_returns = returns @ weights
        daily_return = np.mean(port_returns)
        daily_vol = np.std(port_returns, ddof=1)  # Use ddof=1 for sample standard deviation
        
        # For MSR, we need to handle the case where volatility might be very small
        if is_msr and daily_vol < 1e-10:  # If volatility is effectively zero
            daily_sharpe = 0.0
        else:
            daily_sharpe = (daily_return * 252 - rf) / (daily_vol * np.sqrt(252)) if daily_vol != 0 else 0.0
            
        return daily_return, daily_vol, daily_sharpe
    
    # Calculate annualized statistics
    def portfolio_annual_stats(daily_return, daily_vol, is_msr=False):
        annual_return = daily_return * 252
        annual_vol = daily_vol * np.sqrt(252)
        
        # For MSR, handle the case where volatility might be very small
        if is_msr and annual_vol < 1e-8:  # If annualized volatility is effectively zero
            annual_sharpe = 0.0
        else:
            annual_sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else 0.0
            
        return annual_return, annual_vol, annual_sharpe
    
    # Calculate daily stats
    g_daily_return, g_daily_vol, g_daily_sharpe = portfolio_daily_stats(gmv)
    m_daily_return, m_daily_vol, m_daily_sharpe = portfolio_daily_stats(msr, is_msr=True)
    
    # Calculate annualized stats
    g_annual_return, g_annual_vol, g_annual_sharpe = portfolio_annual_stats(g_daily_return, g_daily_vol)
    m_annual_return, m_annual_vol, m_annual_sharpe = portfolio_annual_stats(m_daily_return, m_daily_vol, is_msr=True)
    
    # Debug prints
    print("\n=== Portfolio Statistics ===")
    print("GMV Daily - Return:", g_daily_return, "Vol:", g_daily_vol, "Sharpe:", g_daily_sharpe)
    print("MSR Daily - Return:", m_daily_return, "Vol:", m_daily_vol, "Sharpe:", m_daily_sharpe)
    print("GMV Annual - Return:", g_annual_return, "Vol:", g_annual_vol, "Sharpe:", g_annual_sharpe)
    print("MSR Annual - Return:", m_annual_return, "Vol:", m_annual_vol, "Sharpe:", m_annual_sharpe)

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(settings.BASE_DIR, 'portfolio', 'static', 'optimizer_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"frontier_{timestamp}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    # Calculate efficient frontier points
    # Build a stable annual return grid and call the unified efficient_frontier
    # annualized expected returns (fractions)
    mu = returns.mean().values * 252.0  

    # build clean target grid from annual return range
    target_grid = np.linspace(float(mu.min()), float(mu.max()), 80)

    # compute smooth frontier
    frontier_raw = efficient_frontier(
    returns, 
    num_points=len(target_grid), 
    target_returns=target_grid
)


    # if solver failed → return empty list
    if not frontier_raw:
        frontier_raw = []

    # convert to clean float dicts (NO fallback, NO synthetic)
    frontier_points = [
        {"x": float(p["x"]), "y": float(p["y"])}
        for p in frontier_raw
    ]

    # Prepare assets data (safe and numeric)
    assets_ready = []
    try:
        # Build assets list from returns (annualized %)
        for ticker in tickers:
            if ticker in returns.columns:
                try:
                    asset_return = float(returns[ticker].mean() * 252 * 100)  # %
                    asset_vol = float(returns[ticker].std(ddof=1) * np.sqrt(252) * 100)  # %
                    assets_ready.append({"x": float(asset_vol), "y": float(asset_return), "ticker": ticker})
                except Exception:
                    continue
    except Exception:
        assets_ready = []

    # Convert weights to dictionary with tickers as keys (safe floats)
    try:
        gmv_weights = {ticker: float(weight) for ticker, weight in zip(tickers, gmv)}
        # If weights appear to be in percent units (e.g., 27.0), convert to decimals
        if any((w is not None) and (w > 1.0 + 1e-8) for w in gmv_weights.values()):
            gmv_weights = {t: float(w) / 100.0 for t, w in gmv_weights.items()}
        # Normalize to sum to 1 if possible
        try:
            total = sum(gmv_weights.values())
            if total > 0:
                gmv_weights = {t: float(w) / total for t, w in gmv_weights.items()}
        except Exception:
            pass
    except Exception:
        gmv_weights = {ticker: 0.0 for ticker in tickers}
    try:
        msr_weights = {ticker: float(weight) for ticker, weight in zip(tickers, msr)}
        # If weights appear to be in percent units (e.g., 27.0), convert to decimals
        if any((w is not None) and (w > 1.0 + 1e-8) for w in msr_weights.values()):
            msr_weights = {t: float(w) / 100.0 for t, w in msr_weights.items()}
        # Normalize to sum to 1 if possible
        try:
            total = sum(msr_weights.values())
            if total > 0:
                msr_weights = {t: float(w) / total for t, w in msr_weights.items()}
        except Exception:
            pass
    except Exception:
        msr_weights = {ticker: 0.0 for ticker in tickers}

    # Combine stats for the template (daily and annual) as percentages where appropriate
    daily_stats = {
        "gmv_return": float(g_daily_return * 100),
        "gmv_volatility": float(g_daily_vol * 100),
        "gmv_sharpe_ratio": float(g_daily_sharpe),
        "msr_return": float(m_daily_return * 100),
        "msr_volatility": float(m_daily_vol * 100),
        "msr_sharpe_ratio": float(m_daily_sharpe),
    }

    annual_stats = {
        "gmv_return": float(g_annual_return * 100),
        "gmv_volatility": float(g_annual_vol * 100),
        "gmv_sharpe_ratio": float(g_annual_sharpe),
        "msr_return": float(m_annual_return * 100),
        "msr_volatility": float(m_annual_vol * 100),
        "msr_sharpe_ratio": float(m_annual_sharpe),
    }

    # Correlation matrices
    try:
        corr_matrix = returns.corr().round(4)
        corr_abs = corr_matrix.abs().round(4)
    except Exception:
        corr_matrix = pd.DataFrame()
        corr_abs = pd.DataFrame()

    # Plot relative path (may be empty if static plot generation removed)
    plot_relative_path = ''

    # Build safe JSON strings (robust to missing assets_ready)
    safe_frontier = json.dumps([
        {"x": float(p["x"]), "y": float(p["y"]) }
        for p in frontier_points
    ])

    try:
        safe_assets = json.dumps([
            {"x": float(p["x"]), "y": float(p["y"]) }
            for p in assets_ready
        ])
    except Exception:
        safe_assets = json.dumps([])

    context = {
        "tickers": tickers,
        "gmv": gmv_weights,
        "msr": msr_weights,
        "daily": daily_stats,
        "annual": annual_stats,
        "corr_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
        "corr_abs": corr_abs.to_dict() if not corr_abs.empty else {},  # Add absolute correlations for coloring
        "start_date": start,
        "end_date": end,
        "risk_free_rate": float(risk_free),
        "frontier_plot": plot_relative_path,
        # Use safe JSON strings for the template consumption
        "efficient_frontier": safe_frontier,
        "assets": safe_assets,
        "safe_frontier": safe_frontier,
        "safe_assets": safe_assets,
    }

    # Debug: log sizes and head of data sent to template
    try:
        print('\n[DEBUG] frontier_points count:', len(frontier_points))
        print('[DEBUG] frontier_points sample:', frontier_points[:5])
    except Exception:
        print('[DEBUG] frontier_points: (error accessing)')
    try:
        parsed_assets = json.loads(safe_assets)
        print('[DEBUG] assets_ready count:', len(parsed_assets))
        print('[DEBUG] assets_ready sample:', parsed_assets[:5])
    except Exception:
        print('[DEBUG] assets_ready: (error accessing)')
    
    return render(request, "optimizer/classic_result.html", context)




# ----------------------------
# RISK PARITY PLACEHOLDER
# ----------------------------
def risk_parity_optimizer(request):
    return render(request, "optimizer/risk_parity_input.html")


# ----------------------------
# HRP PLACEHOLDER
# ----------------------------
def hrp_optimizer(request):
    return render(request, "optimizer/hrp_input.html")
