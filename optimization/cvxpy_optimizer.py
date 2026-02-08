"""
CVXPy-based Convex Portfolio Optimization
Uses convex optimization to maximize risk-adjusted returns
"""

import numpy as np
import pandas as pd
import cvxpy as cp


def optimize_risk_adjusted_return(avg_returns, cov_matrix, gamma_value):
    """
    Optimize portfolio for risk-adjusted return
    
    Objective: max w^T * mu - gamma * w^T * Sigma * w
    
    Parameters:
    -----------
    avg_returns : np.array
        Expected returns
    cov_matrix : np.array
        Covariance matrix
    gamma_value : float
        Risk aversion parameter
    
    Returns:
    --------
    result : dict
        Optimal portfolio with weights, return, and volatility
    """
    n_assets = len(avg_returns)
    
    # Convert to numpy arrays
    if isinstance(avg_returns, pd.Series):
        avg_returns = avg_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Define optimization variables
    weights = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)
    gamma.value = gamma_value
    
    # Portfolio return and volatility
    portfolio_return = avg_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective: maximize risk-adjusted return
    objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1,  # Weights sum to 1
        weights >= 0            # No short selling
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Extract results
    optimal_weights = weights.value
    opt_return = portfolio_return.value
    opt_volatility = cp.sqrt(portfolio_variance).value
    
    return {
        'weights': optimal_weights,
        'return': opt_return,
        'volatility': opt_volatility,
        'sharpe_ratio': opt_return / opt_volatility if opt_volatility > 0 else 0
    }


def efficient_frontier_cvxpy(avg_returns, cov_matrix, n_points=25):
    """
    Calculate efficient frontier using CVXPy
    
    Parameters:
    -----------
    avg_returns : np.array or pd.Series
        Expected returns
    cov_matrix : np.array or pd.DataFrame
        Covariance matrix
    n_points : int
        Number of points on efficient frontier
    
    Returns:
    --------
    results : dict
        Dictionary with returns, volatilities, and weights
    """
    n_assets = len(avg_returns)
    
    # Convert to numpy arrays
    if isinstance(avg_returns, pd.Series):
        avg_returns = avg_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Define optimization variables
    weights = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)
    
    # Portfolio metrics
    portfolio_return = avg_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective
    objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)
    
    # Constraints
    constraints = [cp.sum(weights) == 1, weights >= 0]
    
    # Problem
    problem = cp.Problem(objective, constraints)
    
    # Calculate efficient frontier for different gamma values
    gamma_range = np.logspace(-3, 3, num=n_points)
    
    ef_returns = np.zeros(n_points)
    ef_volatility = np.zeros(n_points)
    ef_weights = []
    
    for i in range(n_points):
        gamma.value = gamma_range[i]
        problem.solve()
        ef_volatility[i] = cp.sqrt(portfolio_variance).value
        ef_returns[i] = portfolio_return.value
        ef_weights.append(weights.value)
    
    return {
        'returns': ef_returns,
        'volatility': ef_volatility,
        'weights': ef_weights,
        'gamma_range': gamma_range
    }


def optimize_with_leverage(avg_returns, cov_matrix, max_leverage_value, n_points=25):
    """
    Optimize portfolio with leverage constraints
    
    Parameters:
    -----------
    avg_returns : np.array or pd.Series
        Expected returns
    cov_matrix : np.array or pd.DataFrame
        Covariance matrix
    max_leverage_value : float
        Maximum leverage (L1 norm of weights)
    n_points : int
        Number of gamma values
    
    Returns:
    --------
    results : dict
        Efficient frontier with leverage constraint
    """
    n_assets = len(avg_returns)
    
    # Convert to numpy arrays
    if isinstance(avg_returns, pd.Series):
        avg_returns = avg_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # Define variables and parameters
    weights = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)
    max_leverage = cp.Parameter()
    max_leverage.value = max_leverage_value
    
    # Portfolio metrics
    portfolio_return = avg_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective
    objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)
    
    # Constraints (with leverage)
    constraints = [
        cp.sum(weights) == 1,
        cp.norm(weights, 1) <= max_leverage
    ]
    
    # Problem
    problem = cp.Problem(objective, constraints)
    
    # Solve for different gamma values
    gamma_range = np.logspace(-3, 3, num=n_points)
    
    ef_returns = np.zeros(n_points)
    ef_volatility = np.zeros(n_points)
    ef_weights = []
    
    for i in range(n_points):
        gamma.value = gamma_range[i]
        problem.solve()
        ef_volatility[i] = cp.sqrt(portfolio_variance).value
        ef_returns[i] = portfolio_return.value
        ef_weights.append(weights.value)
    
    return {
        'returns': ef_returns,
        'volatility': ef_volatility,
        'weights': ef_weights,
        'gamma_range': gamma_range
    }


def compare_leverage_levels(avg_returns, cov_matrix, leverage_levels, n_points=25):
    """
    Compare efficient frontiers for different leverage levels
    
    Parameters:
    -----------
    avg_returns : np.array or pd.Series
        Expected returns
    cov_matrix : np.array or pd.DataFrame
        Covariance matrix
    leverage_levels : list
        List of leverage levels to compare
    n_points : int
        Number of points per frontier
    
    Returns:
    --------
    results : dict
        Results for each leverage level
    """
    all_results = {}
    
    for leverage in leverage_levels:
        result = optimize_with_leverage(avg_returns, cov_matrix, leverage, n_points)
        all_results[leverage] = result
    
    return all_results
