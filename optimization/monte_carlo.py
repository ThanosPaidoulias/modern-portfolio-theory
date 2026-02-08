"""
Monte Carlo Portfolio Optimization
Simulates random portfolios to find efficient frontier
"""

import numpy as np
import pandas as pd
from src.portfolio_metrics import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_sharpe_ratio
)


def monte_carlo_simulation(avg_returns, cov_matrix, n_portfolios=100000, rf_rate=0, seed=42):
    """
    Generate random portfolios using Monte Carlo simulation
    
    Parameters:
    -----------
    avg_returns : pd.Series or np.array
        Expected returns for each asset
    cov_matrix : pd.DataFrame or np.array
        Covariance matrix
    n_portfolios : int
        Number of portfolios to simulate
    rf_rate : float
        Risk-free rate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with returns, volatility, and Sharpe ratio for each portfolio
    all_weights : np.array
        Array of all portfolio weights
    """
    n_assets = len(avg_returns)
    
    # Set random seed
    np.random.seed(seed)
    
    # Generate random weights
    weights = np.random.random(size=(n_portfolios, n_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    
    # Calculate portfolio metrics
    portf_returns = np.dot(weights, avg_returns)
    
    portf_volatility = []
    for i in range(len(weights)):
        vol = np.sqrt(np.dot(weights[i].T, np.dot(cov_matrix, weights[i])))
        portf_volatility.append(vol)
    portf_volatility = np.array(portf_volatility)
    
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'returns': portf_returns,
        'volatility': portf_volatility,
        'sharpe_ratio': portf_sharpe_ratio
    })
    
    return results_df, weights


def find_efficient_frontier_mc(results_df, weights, n_points=100):
    """
    Find efficient frontier from Monte Carlo simulation results
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from Monte Carlo simulation
    weights : np.array
        Portfolio weights from simulation
    n_points : int
        Number of points on efficient frontier
    
    Returns:
    --------
    ef_returns : np.array
        Returns on efficient frontier
    ef_volatility : np.array
        Volatility on efficient frontier
    """
    portf_returns = results_df['returns'].values
    portf_volatility = results_df['volatility'].values
    
    # Create range of returns
    ef_returns = np.linspace(
        results_df.returns.min(),
        results_df.returns.max(),
        n_points
    )
    ef_returns = np.round(ef_returns, 2)
    portf_returns_rounded = np.round(portf_returns, 2)
    
    # Find minimum volatility for each return level
    ef_volatility = []
    indices_to_skip = []
    
    for point_index in range(n_points):
        if ef_returns[point_index] not in portf_returns_rounded:
            indices_to_skip.append(point_index)
            continue
        matched_ind = np.where(portf_returns_rounded == ef_returns[point_index])
        ef_volatility.append(np.min(portf_volatility[matched_ind]))
    
    ef_returns = np.delete(ef_returns, indices_to_skip)
    
    return ef_returns, np.array(ef_volatility)


def find_optimal_portfolios(results_df, weights):
    """
    Find maximum Sharpe ratio and minimum volatility portfolios
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from Monte Carlo simulation
    weights : np.array
        Portfolio weights
    
    Returns:
    --------
    optimal_portfolios : dict
        Dictionary with max Sharpe and min volatility portfolios
    """
    # Maximum Sharpe ratio portfolio
    max_sharpe_idx = np.argmax(results_df.sharpe_ratio)
    max_sharpe_portfolio = {
        'metrics': results_df.loc[max_sharpe_idx],
        'weights': weights[max_sharpe_idx]
    }
    
    # Minimum volatility portfolio
    min_vol_idx = np.argmin(results_df.volatility)
    min_vol_portfolio = {
        'metrics': results_df.loc[min_vol_idx],
        'weights': weights[min_vol_idx]
    }
    
    return {
        'max_sharpe': max_sharpe_portfolio,
        'min_volatility': min_vol_portfolio
    }
