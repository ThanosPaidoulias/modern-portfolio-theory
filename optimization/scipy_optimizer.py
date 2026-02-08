"""
SciPy-based Portfolio Optimization
Uses Sequential Least Squares Programming (SLSQP) for optimization
"""

import numpy as np
import scipy.optimize as sco


def get_portfolio_return(weights, avg_returns):
    """Calculate portfolio return"""
    return np.sum(avg_returns * weights)


def get_portfolio_volatility(weights, avg_returns, cov_matrix):
    """Calculate portfolio volatility"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def neg_sharpe_ratio(weights, avg_returns, cov_matrix, rf_rate):
    """
    Calculate negative Sharpe ratio (for minimization)
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    avg_returns : np.array
        Expected returns
    cov_matrix : np.array
        Covariance matrix
    rf_rate : float
        Risk-free rate
    
    Returns:
    --------
    neg_sharpe : float
        Negative Sharpe ratio
    """
    portfolio_return = np.sum(avg_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
    return -sharpe_ratio


def minimize_volatility(avg_returns, cov_matrix, target_return):
    """
    Find portfolio with minimum volatility for a target return
    
    Parameters:
    -----------
    avg_returns : np.array
        Expected returns
    cov_matrix : np.array
        Covariance matrix
    target_return : float
        Target portfolio return
    
    Returns:
    --------
    result : scipy.optimize.OptimizeResult
        Optimization result
    """
    n_assets = len(avg_returns)
    args = (avg_returns, cov_matrix)
    
    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: get_portfolio_return(x, avg_returns) - target_return},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    
    # Bounds (weights between 0 and 1, no short selling)
    bounds = tuple((0, 1) for asset in range(n_assets))
    
    # Initial guess (equal weights)
    initial_guess = n_assets * [1.0 / n_assets]
    
    # Optimize
    result = sco.minimize(
        get_portfolio_volatility,
        initial_guess,
        args=args,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    
    return result


def get_efficient_frontier_scipy(avg_returns, cov_matrix, returns_range):
    """
    Calculate efficient frontier using scipy optimization
    
    Parameters:
    -----------
    avg_returns : np.array
        Expected returns
    cov_matrix : np.array
        Covariance matrix
    returns_range : np.array
        Range of target returns
    
    Returns:
    --------
    efficient_portfolios : list
        List of optimization results for each target return
    """
    efficient_portfolios = []
    
    for target_return in returns_range:
        result = minimize_volatility(avg_returns, cov_matrix, target_return)
        efficient_portfolios.append(result)
    
    return efficient_portfolios


def maximize_sharpe_ratio(avg_returns, cov_matrix, rf_rate=0):
    """
    Find portfolio with maximum Sharpe ratio
    
    Parameters:
    -----------
    avg_returns : np.array
        Expected returns
    cov_matrix : np.array
        Covariance matrix
    rf_rate : float
        Risk-free rate
    
    Returns:
    --------
    result : dict
        Optimal portfolio with weights and metrics
    """
    n_assets = len(avg_returns)
    args = (avg_returns, cov_matrix, rf_rate)
    
    # Constraints (weights sum to 1)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for asset in range(n_assets))
    
    # Initial guess
    initial_guess = n_assets * [1.0 / n_assets]
    
    # Optimize (minimize negative Sharpe ratio = maximize Sharpe ratio)
    opt_result = sco.minimize(
        neg_sharpe_ratio,
        x0=initial_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract results
    optimal_weights = opt_result['x']
    portfolio_return = get_portfolio_return(optimal_weights, avg_returns)
    portfolio_vol = get_portfolio_volatility(optimal_weights, avg_returns, cov_matrix)
    sharpe = (portfolio_return - rf_rate) / portfolio_vol
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe
    }


def find_minimum_volatility_portfolio(avg_returns, cov_matrix):
    """
    Find global minimum volatility portfolio
    
    Parameters:
    -----------
    avg_returns : np.array
        Expected returns
    cov_matrix : np.array
        Covariance matrix
    
    Returns:
    --------
    result : dict
        Minimum volatility portfolio with weights and metrics
    """
    # Find efficient frontier
    returns_range = np.linspace(0.02, avg_returns.max(), 200)
    efficient_portfolios = get_efficient_frontier_scipy(avg_returns, cov_matrix, returns_range)
    
    # Extract volatilities
    volatilities = [port['fun'] for port in efficient_portfolios]
    
    # Find minimum
    min_vol_idx = np.argmin(volatilities)
    min_vol_portfolio = efficient_portfolios[min_vol_idx]
    
    return {
        'weights': min_vol_portfolio['x'],
        'return': returns_range[min_vol_idx],
        'volatility': min_vol_portfolio['fun'],
        'sharpe_ratio': returns_range[min_vol_idx] / min_vol_portfolio['fun']
    }
