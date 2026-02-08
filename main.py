#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Portfolio Theory - Complete Implementation
Four optimization approaches to portfolio construction

Author: Thanos Paidoulias
Date: 2021 (Refactored 2026)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loader import (
    download_stock_data,
    calculate_returns,
    annualize_statistics,
    get_correlation_matrix
)
from src.portfolio_metrics import (
    equal_weight_portfolio,
    calculate_portfolio_metrics
)
from src.optimization.monte_carlo import (
    monte_carlo_simulation,
    find_efficient_frontier_mc,
    find_optimal_portfolios
)
from src.optimization.scipy_optimizer import (
    get_efficient_frontier_scipy,
    maximize_sharpe_ratio,
    find_minimum_volatility_portfolio
)
from src.optimization.cvxpy_optimizer import (
    efficient_frontier_cvxpy,
    compare_leverage_levels
)
from src.visualization import (
    plot_price_history,
    plot_returns_histogram,
    plot_correlation_heatmap,
    plot_daily_returns,
    plot_efficient_frontier,
    plot_optimal_portfolios,
    plot_weight_allocation,
    plot_weights_by_risk_aversion,
    compare_efficient_frontiers,
    save_figure
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Stock selection
RISKY_ASSETS = ['AAPL', 'TSLA', 'DAL', 'PFE']
START_DATE = '2019-01-01'
END_DATE = '2020-12-31'

# Parameters
N_PORTFOLIOS = 100000  # Monte Carlo simulations
N_DAYS = 252           # Trading days per year
RF_RATE = 0            # Risk-free rate
SEED = 42              # Random seed for reproducibility

# Asset info
ASSET_INFO = {
    'AAPL': 'Apple - Stable tech giant',
    'TSLA': 'Tesla - High volatility growth',
    'DAL': 'Delta Airlines - Traditional industry',
    'PFE': 'Pfizer - Pharmaceutical'
}


def print_portfolio_summary(name, metrics, weights, asset_names):
    """Print portfolio summary nicely"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print("\nPerformance Metrics:")
    print(f"  Return:      {metrics['return']*100:>8.2f}%")
    print(f"  Volatility:  {metrics['volatility']*100:>8.2f}%")
    print(f"  Sharpe Ratio:{metrics['sharpe_ratio']:>8.2f}")
    print("\nPortfolio Weights:")
    for asset, weight in zip(asset_names, weights):
        print(f"  {asset:6s}: {weight*100:>6.2f}%")
    print(f"{'='*60}\n")


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("MODERN PORTFOLIO THEORY - PORTFOLIO OPTIMIZATION")
    print("="*70)
    
    # ========================================================================
    # 1. DATA ACQUISITION & PREPARATION
    # ========================================================================
    print("\n[STEP 1] Downloading stock data...")
    prices_df = download_stock_data(RISKY_ASSETS, START_DATE, END_DATE)
    
    print("\n[STEP 1] Calculating returns...")
    returns = calculate_returns(prices_df)
    avg_returns, cov_mat = annualize_statistics(returns, N_DAYS)
    cor_mat = get_correlation_matrix(returns)
    
    n_assets = len(RISKY_ASSETS)
    
    # Display basic statistics
    print("\nAnnualized Returns:")
    for asset, ret in zip(RISKY_ASSETS, avg_returns):
        print(f"  {asset}: {ret*100:.2f}%")
    
    # ========================================================================
    # 2. EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print("\n[STEP 2] Creating visualizations...")
    
    # Price history
    fig1 = plot_price_history(prices_df, 'Stock Prices (2019-2020)')
    save_figure(fig1, 'results/01_price_history.png')
    
    # Returns histogram
    fig2 = plot_returns_histogram(returns)
    save_figure(fig2, 'results/02_returns_distribution.png')
    
    # Correlation heatmap
    fig3 = plot_correlation_heatmap(cor_mat, 'Asset Correlation Matrix')
    save_figure(fig3, 'results/03_correlation_matrix.png')
    
    # Daily returns over time
    fig4 = plot_daily_returns(returns, 'Daily Returns Over Time')
    save_figure(fig4, 'results/04_daily_returns.png')
    
    plt.close('all')  # Close all figures to free memory
    
    # ========================================================================
    # 3. EQUAL WEIGHT (1/n) PORTFOLIO
    # ========================================================================
    print("\n[STEP 3] Analyzing equal-weight (1/n) portfolio...")
    
    equal_weights = equal_weight_portfolio(n_assets)
    equal_metrics = calculate_portfolio_metrics(equal_weights, avg_returns, cov_mat, RF_RATE)
    
    print_portfolio_summary(
        "EQUAL WEIGHT (1/n) PORTFOLIO",
        equal_metrics,
        equal_weights,
        RISKY_ASSETS
    )
    
    # ========================================================================
    # 4. MONTE CARLO SIMULATION
    # ========================================================================
    print("\n[STEP 4] Running Monte Carlo simulation...")
    print(f"  Simulating {N_PORTFOLIOS:,} random portfolios...")
    
    mc_results, mc_weights = monte_carlo_simulation(
        avg_returns, cov_mat, N_PORTFOLIOS, RF_RATE, SEED
    )
    
    # Find efficient frontier
    ef_returns_mc, ef_vol_mc = find_efficient_frontier_mc(mc_results, mc_weights, n_points=100)
    
    # Find optimal portfolios
    optimal_mc = find_optimal_portfolios(mc_results, mc_weights)
    
    # Print results
    max_sharpe_mc = optimal_mc['max_sharpe']
    print_portfolio_summary(
        "MONTE CARLO - MAXIMUM SHARPE RATIO",
        max_sharpe_mc['metrics'],
        max_sharpe_mc['weights'],
        RISKY_ASSETS
    )
    
    min_vol_mc = optimal_mc['min_volatility']
    print_portfolio_summary(
        "MONTE CARLO - MINIMUM VOLATILITY",
        min_vol_mc['metrics'],
        min_vol_mc['weights'],
        RISKY_ASSETS
    )
    
    # Visualizations
    fig5 = plot_efficient_frontier(
        mc_results,
        ef_returns_mc,
        ef_vol_mc,
        avg_returns,
        RISKY_ASSETS,
        cov_mat
    )
    save_figure(fig5, 'results/05_efficient_frontier_mc.png')
    
    fig6 = plot_optimal_portfolios(mc_results, max_sharpe_mc, min_vol_mc)
    save_figure(fig6, 'results/06_optimal_portfolios_mc.png')
    
    plt.close('all')
    
    # ========================================================================
    # 5. SCIPY OPTIMIZATION
    # ========================================================================
    print("\n[STEP 5] Running SciPy optimization (SLSQP)...")
    
    # Maximum Sharpe ratio
    max_sharpe_scipy = maximize_sharpe_ratio(avg_returns.values, cov_mat.values, RF_RATE)
    print_portfolio_summary(
        "SCIPY OPTIMIZATION - MAXIMUM SHARPE RATIO",
        max_sharpe_scipy,
        max_sharpe_scipy['weights'],
        RISKY_ASSETS
    )
    
    # Minimum volatility
    min_vol_scipy = find_minimum_volatility_portfolio(avg_returns.values, cov_mat.values)
    print_portfolio_summary(
        "SCIPY OPTIMIZATION - MINIMUM VOLATILITY",
        min_vol_scipy,
        min_vol_scipy['weights'],
        RISKY_ASSETS
    )
    
    # Efficient frontier
    returns_range = np.linspace(0.02, 1.4, 200)
    efficient_portfolios_scipy = get_efficient_frontier_scipy(
        avg_returns.values,
        cov_mat.values,
        returns_range
    )
    vols_range_scipy = [p['fun'] for p in efficient_portfolios_scipy]
    
    # Visualizations
    fig7 = plot_efficient_frontier(
        mc_results,
        returns_range,
        vols_range_scipy,
        avg_returns,
        RISKY_ASSETS,
        cov_mat
    )
    save_figure(fig7, 'results/07_efficient_frontier_scipy.png')
    
    fig8 = plot_weight_allocation(
        max_sharpe_scipy['weights'],
        RISKY_ASSETS,
        'SciPy Max Sharpe - Weight Allocation'
    )
    save_figure(fig8, 'results/08_weights_scipy_max_sharpe.png')
    
    plt.close('all')
    
    # ========================================================================
    # 6. CVXPY CONVEX OPTIMIZATION
    # ========================================================================
    print("\n[STEP 6] Running CVXPy convex optimization...")
    
    # Efficient frontier with different risk aversions
    ef_cvxpy = efficient_frontier_cvxpy(avg_returns, cov_mat, n_points=25)
    
    print(f"\nGenerated efficient frontier with {len(ef_cvxpy['returns'])} points")
    print(f"Risk aversion range: γ = {ef_cvxpy['gamma_range'].min():.3f} to {ef_cvxpy['gamma_range'].max():.3f}")
    
    # Visualizations
    fig9 = plot_weights_by_risk_aversion(
        np.array(ef_cvxpy['weights']),
        RISKY_ASSETS,
        ef_cvxpy['gamma_range']
    )
    save_figure(fig9, 'results/09_weights_by_risk_aversion.png')
    
    # Compare different leverage levels
    print("\n[STEP 6b] Analyzing leverage constraints...")
    leverage_levels = [1, 3, 5]
    leverage_results = compare_leverage_levels(
        avg_returns,
        cov_mat,
        leverage_levels,
        n_points=25
    )
    
    # Plot comparison
    frontiers_dict = {
        f'Leverage = {lev}': {
            'volatility': leverage_results[lev]['volatility'],
            'returns': leverage_results[lev]['returns']
        }
        for lev in leverage_levels
    }
    
    fig10 = compare_efficient_frontiers(
        frontiers_dict,
        'Efficient Frontier - Different Leverage Levels'
    )
    save_figure(fig10, 'results/10_leverage_comparison.png')
    
    plt.close('all')
    
    # ========================================================================
    # 7. COMPARISON OF METHODS
    # ========================================================================
    print("\n[STEP 7] Comparing all optimization methods...")
    
    # Create comparison DataFrame
    comparison_data = {
        'Method': [
            'Equal Weight (1/n)',
            'Monte Carlo - Max Sharpe',
            'Monte Carlo - Min Vol',
            'SciPy - Max Sharpe',
            'SciPy - Min Vol'
        ],
        'Return (%)': [
            equal_metrics['return'] * 100,
            max_sharpe_mc['metrics']['returns'] * 100,
            min_vol_mc['metrics']['returns'] * 100,
            max_sharpe_scipy['return'] * 100,
            min_vol_scipy['return'] * 100
        ],
        'Volatility (%)': [
            equal_metrics['volatility'] * 100,
            max_sharpe_mc['metrics']['volatility'] * 100,
            min_vol_mc['metrics']['volatility'] * 100,
            max_sharpe_scipy['volatility'] * 100,
            min_vol_scipy['volatility'] * 100
        ],
        'Sharpe Ratio': [
            equal_metrics['sharpe_ratio'],
            max_sharpe_mc['metrics']['sharpe_ratio'],
            min_vol_mc['metrics']['sharpe_ratio'],
            max_sharpe_scipy['sharpe_ratio'],
            min_vol_scipy['sharpe_ratio']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("SUMMARY: COMPARISON OF ALL METHODS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Save comparison table
    comparison_df.to_csv('results/portfolio_comparison.csv', index=False)
    print("\nComparison table saved to: results/portfolio_comparison.csv")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nAll results saved to 'results/' directory:")
    print("  - Visualizations (10 figures)")
    print("  - Portfolio comparison table")
    print("\nKey Findings:")
    print(f"  Best Sharpe Ratio: {comparison_df['Sharpe Ratio'].max():.2f} ({comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Method']})")
    print(f"  Lowest Volatility: {comparison_df['Volatility (%)'].min():.2f}% ({comparison_df.loc[comparison_df['Volatility (%)'].idxmin(), 'Method']})")
    print(f"  Highest Return:    {comparison_df['Return (%)'].max():.2f}% ({comparison_df.loc[comparison_df['Return (%)'].idxmax(), 'Method']})")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
