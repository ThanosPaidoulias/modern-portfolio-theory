"""
Portfolio Metrics Calculation Module
Functions for calculating portfolio returns, volatility, and Sharpe ratio
"""

import numpy as np
import pandas as pd


def calculate_portfolio_return(weights, avg_returns):
    """
    Calculate expected portfolio return
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    avg_returns : np.array or pd.Series
        Expected returns for each asset
    
    Returns:
    --------
    portfolio_return : float
        Expected portfolio return
    """
    return np.sum(avg_returns * weights)


def calculate_portfolio_volatility(weights, cov_matrix):
    """
    Calculate portfolio volatility (standard deviation)
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    cov_matrix : np.array or pd.DataFrame
        Covariance matrix of asset returns
    
    Returns:
    --------
    portfolio_vol : float
        Portfolio volatility
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def calculate_sharpe_ratio(portfolio_return, portfolio_volatility, risk_free_rate=0):
    """
    Calculate Sharpe ratio
    
    Parameters:
    -----------
    portfolio_return : float
        Expected portfolio return
    portfolio_volatility : float
        Portfolio volatility
    risk_free_rate : float
        Risk-free rate (default: 0)
    
    Returns:
    --------
    sharpe_ratio : float
        Sharpe ratio
    """
    return (portfolio_return - risk_free_rate) / portfolio_volatility


def equal_weight_portfolio(n_assets):
    """
    Create equal-weighted (1/n) portfolio
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    
    Returns:
    --------
    weights : np.array
        Equal weights for all assets
    """
    return np.array([1.0 / n_assets] * n_assets)


def calculate_portfolio_metrics(weights, avg_returns, cov_matrix, rf_rate=0):
    """
    Calculate all portfolio metrics at once
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    avg_returns : np.array or pd.Series
        Expected returns
    cov_matrix : np.array or pd.DataFrame
        Covariance matrix
    rf_rate : float
        Risk-free rate
    
    Returns:
    --------
    metrics : dict
        Dictionary with return, volatility, and Sharpe ratio
    """
    portfolio_return = calculate_portfolio_return(weights, avg_returns)
    portfolio_vol = calculate_portfolio_volatility(weights, cov_matrix)
    sharpe = calculate_sharpe_ratio(portfolio_return, portfolio_vol, rf_rate)
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe
    }
