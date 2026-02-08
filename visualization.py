"""
Visualization Module for Portfolio Optimization
Functions for creating plots and charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_price_history(prices_df, title='Stock Prices'):
    """
    Plot historical stock prices
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    prices_df['Adj Close'].plot(ax=ax, title=title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='best')
    plt.tight_layout()
    return fig


def plot_returns_histogram(returns, bins=40, figsize=(12, 8)):
    """
    Plot histogram of returns for each asset
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    """
    fig = returns.hist(bins=bins, figsize=figsize)
    plt.suptitle('Distribution of Daily Returns', y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_matrix, title='Correlation Matrix'):
    """
    Plot correlation heatmap
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.zeros_like(correlation_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    # Plot heatmap
    sns.heatmap(
        correlation_matrix,
        cmap='RdYlGn',
        vmax=1.0,
        vmin=-1.0,
        mask=mask,
        linewidths=2.5,
        annot=True,
        fmt='.2f',
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_covariance_heatmap(covariance_matrix, title='Covariance Matrix'):
    """
    Plot covariance heatmap
    
    Parameters:
    -----------
    covariance_matrix : pd.DataFrame
        Covariance matrix
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.zeros_like(covariance_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    # Plot heatmap
    sns.heatmap(
        covariance_matrix,
        cmap='RdYlGn',
        mask=mask,
        linewidths=2.5,
        annot=True,
        fmt='g',
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_daily_returns(returns, title='Daily Returns Over Time'):
    """
    Plot daily returns time series
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for column in returns.columns:
        ax.plot(returns.index, returns[column], lw=2, alpha=0.8, label=column)
    
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylabel('Daily Returns')
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_efficient_frontier(results_df, ef_returns=None, ef_volatility=None, 
                            individual_assets=None, asset_names=None, cov_matrix=None):
    """
    Plot efficient frontier with Monte Carlo portfolios
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Monte Carlo simulation results
    ef_returns : np.array
        Efficient frontier returns
    ef_volatility : np.array
        Efficient frontier volatility
    individual_assets : np.array
        Individual asset returns
    asset_names : list
        Asset names
    cov_matrix : np.array
        Covariance matrix (for individual asset volatility)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all portfolios
    scatter = ax.scatter(
        results_df['volatility'],
        results_df['returns'],
        c=results_df['sharpe_ratio'],
        cmap='RdYlGn',
        edgecolors='black',
        alpha=0.7,
        s=20
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
    
    # Plot efficient frontier
    if ef_returns is not None and ef_volatility is not None:
        ax.plot(ef_volatility, ef_returns, 'b--', linewidth=3, label='Efficient Frontier')
    
    # Plot individual assets
    if individual_assets is not None and cov_matrix is not None and asset_names is not None:
        markers = ['o', 'X', 'd', '*', 's', '^', 'v', 'P']
        for i, asset in enumerate(asset_names):
            asset_vol = np.sqrt(cov_matrix.iloc[i, i] if hasattr(cov_matrix, 'iloc') else cov_matrix[i, i])
            asset_return = individual_assets[i]
            ax.scatter(
                asset_vol,
                asset_return,
                marker=markers[i % len(markers)],
                s=200,
                color='black',
                label=asset,
                zorder=5
            )
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier - Portfolio Optimization', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_optimal_portfolios(results_df, max_sharpe_portfolio, min_vol_portfolio):
    """
    Plot efficient frontier highlighting optimal portfolios
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Monte Carlo results
    max_sharpe_portfolio : dict
        Maximum Sharpe ratio portfolio
    min_vol_portfolio : dict
        Minimum volatility portfolio
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all portfolios
    scatter = ax.scatter(
        results_df['volatility'],
        results_df['returns'],
        c=results_df['sharpe_ratio'],
        cmap='RdYlGn',
        edgecolors='black',
        alpha=0.7,
        s=20
    )
    
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    # Highlight maximum Sharpe ratio portfolio
    max_sharpe_metrics = max_sharpe_portfolio['metrics']
    ax.scatter(
        max_sharpe_metrics['volatility'],
        max_sharpe_metrics['returns'],
        c='black',
        marker='*',
        s=500,
        label='Max Sharpe Ratio',
        edgecolors='yellow',
        linewidths=2,
        zorder=5
    )
    
    # Highlight minimum volatility portfolio
    min_vol_metrics = min_vol_portfolio['metrics']
    ax.scatter(
        min_vol_metrics['volatility'],
        min_vol_metrics['returns'],
        c='black',
        marker='P',
        s=400,
        label='Min Volatility',
        edgecolors='blue',
        linewidths=2,
        zorder=5
    )
    
    ax.set_xlabel('Volatility', fontsize=12)
    ax.set_ylabel('Expected Returns', fontsize=12)
    ax.set_title('Optimal Portfolios', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_weight_allocation(weights, asset_names, title='Portfolio Weight Allocation'):
    """
    Plot portfolio weight allocation as bar chart
    
    Parameters:
    -----------
    weights : np.array or dict
        Portfolio weights
    asset_names : list
        Asset names
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(weights, dict):
        weights = list(weights.values())
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(asset_names)))
    
    bars = ax.bar(asset_names, weights * 100, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_ylabel('Weight (%)', fontsize=12)
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, max(weights * 100) * 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_weights_by_risk_aversion(weights_array, asset_names, gamma_range):
    """
    Plot stacked bar chart of weights for different risk aversion levels
    
    Parameters:
    -----------
    weights_array : np.array
        Array of weights for different gamma values
    asset_names : list
        Asset names
    gamma_range : np.array
        Range of gamma (risk aversion) values
    """
    weights_df = pd.DataFrame(
        weights_array,
        columns=asset_names,
        index=np.round(gamma_range, 3)
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    weights_df.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn')
    
    ax.set_title('Weight Allocation per Risk-Aversion Level', fontsize=14)
    ax.set_xlabel('Risk Aversion (γ)', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig


def compare_efficient_frontiers(frontiers_dict, title='Efficient Frontier Comparison'):
    """
    Compare multiple efficient frontiers on one plot
    
    Parameters:
    -----------
    frontiers_dict : dict
        Dictionary with frontier data {label: {'volatility': [...], 'returns': [...]}}
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(frontiers_dict)))
    
    for (label, data), color in zip(frontiers_dict.items(), colors):
        ax.plot(
            data['volatility'],
            data['returns'],
            label=label,
            linewidth=3,
            color=color
        )
    
    ax.set_xlabel('Volatility', fontsize=12)
    ax.set_ylabel('Expected Returns', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def save_figure(fig, filename, dpi=300):
    """
    Save figure to file
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    dpi : int
        Resolution in dots per inch
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filename}")
