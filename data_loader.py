"""
Data Loading Module for Portfolio Optimization
Downloads stock price data from Yahoo Finance and calculates returns
"""

import yfinance as yf
import pandas as pd


def download_stock_data(tickers, start_date, end_date):
    """
    Download historical stock price data from Yahoo Finance
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    prices_df : pd.DataFrame
        DataFrame with adjusted close prices
    """
    prices_df = yf.download(tickers, start=start_date, end=end_date, adjusted=True)
    print(f'Downloaded {prices_df.shape[0]} rows of data.')
    return prices_df


def calculate_returns(prices_df):
    """
    Calculate daily percentage returns from price data
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    
    Returns:
    --------
    returns : pd.DataFrame
        DataFrame with daily returns
    """
    returns = prices_df['Adj Close'].pct_change().dropna()
    return returns


def annualize_statistics(returns, n_days=252):
    """
    Annualize returns and covariance matrix
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    n_days : int
        Number of trading days per year (default: 252)
    
    Returns:
    --------
    avg_returns : pd.Series
        Annualized average returns
    cov_mat : pd.DataFrame
        Annualized covariance matrix
    """
    avg_returns = returns.mean() * n_days
    cov_mat = returns.cov() * n_days
    return avg_returns, cov_mat


def get_correlation_matrix(returns):
    """
    Calculate correlation matrix from returns
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    
    Returns:
    --------
    cor_mat : pd.DataFrame
        Correlation matrix
    """
    return returns.corr()
