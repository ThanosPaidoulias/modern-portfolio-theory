# Modern Portfolio Theory - Python Implementation

**Professional modular implementation of portfolio optimization with 4 different optimization approaches**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

This project implements **Modern Portfolio Theory (MPT)** using Python, demonstrating four different optimization approaches to construct optimal investment portfolios. The codebase is modular, production-ready, and educational.

**Four Optimization Methods:**
1. **Equal Weight (1/n)** - Baseline diversification strategy
2. **Monte Carlo Simulation** - 100,000 random portfolios to find efficient frontier
3. **SciPy Optimization (SLSQP)** - Sequential Least Squares Programming
4. **CVXPy Convex Optimization** - Risk-adjusted return maximization

---

## 📊 Quick Example
```python
from src.data_loader import download_stock_data, calculate_returns, annualize_statistics
from src.optimization.scipy_optimizer import maximize_sharpe_ratio

# Download data
prices = download_stock_data(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2023-01-01')
returns = calculate_returns(prices)
avg_returns, cov_matrix = annualize_statistics(returns)

# Find optimal portfolio
optimal = maximize_sharpe_ratio(avg_returns.values, cov_matrix.values)
print(f"Max Sharpe Ratio: {optimal['sharpe_ratio']:.2f}")
print(f"Weights: {optimal['weights']}")
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/ThanosPaidoulias/modern-portfolio-theory.git
cd modern-portfolio-theory

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Analysis
```bash
python main.py
```

**What happens:**
- Downloads stock data (AAPL, TSLA, DAL, PFE) for 2019-2020
- Calculates returns and covariance matrices
- Runs all 4 optimization methods
- Generates 10+ professional visualizations
- Saves results to `results/` directory
- Prints comprehensive comparison table

**Runtime:** 2-5 minutes

---

## 📂 Project Structure
```
portfolio-optimization/
│
├── main.py                          # Main executable script
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code modules
│   ├── data_loader.py              # Data download & processing
│   ├── portfolio_metrics.py        # Portfolio calculations
│   ├── visualization.py            # Plotting functions
│   └── optimization/               # Optimization algorithms
│       ├── monte_carlo.py          # Monte Carlo simulation
│       ├── scipy_optimizer.py      # SciPy SLSQP optimization
│       └── cvxpy_optimizer.py      # Convex optimization
│
├── original_script                 # Original monolithic implementation
│   └── One_script_version.py
│
├── notebooks/                      # Jupyter notebooks (optional)
├── results/                        # Output plots & tables
└── data/                           # Downloaded data cache
```

**Total:** 1,558 lines of modular, documented code

---

## 📜 Code Versions: Modular vs Original

This project demonstrates the evolution from a research script to production-quality code.

### **Modular Version (Recommended)** ✅
- **Location:** `main.py` + `src/` modules
- **Structure:** 1,558 lines across 8 organized files
- **Best for:** 
  - Production use
  - Code reuse
  - Testing and extension
  - Collaborative development
- **Run:** `python main.py`

**Advantages:**
- ✅ Reusable components
- ✅ Easy to test individual functions
- ✅ Professional architecture
- ✅ Separation of concerns
- ✅ Better for portfolios/interviews

### **Original Script (Reference)** 📄
- **Location:** `original_script/One_script_version.py`
- **Structure:** 491 lines in single file
- **Best for:**
  - Quick exploration
  - Linear understanding of workflow
  - Academic reference
  - Comparison with modular approach
- **Run:** `python original_script/One_script_version.py`

**Note:** Both versions implement the same algorithms and produce equivalent results. The modular version demonstrates software engineering best practices applied to quantitative finance.

---

## 📖 Theory Background

### Modern Portfolio Theory (Markowitz, 1952)

**Core Principle:** Investors can construct portfolios to maximize expected return for a given level of risk by carefully choosing proportions of various assets.

**Key Concepts:**
- **Diversification**: Combining assets with low/negative correlation reduces portfolio risk
- **Efficient Frontier**: Set of optimal portfolios offering highest return for each risk level
- **Sharpe Ratio**: Risk-adjusted return metric = (Return - Risk-Free Rate) / Volatility

### Mathematical Formulation

**Portfolio Return:**
```
R_p = Σ(w_i * R_i)
```

**Portfolio Volatility:**
```
σ_p = √(w^T * Σ * w)
```

**Sharpe Ratio:**
```
SR = (R_p - R_f) / σ_p
```

Where:
- `w` = weight vector
- `R` = expected returns
- `Σ` = covariance matrix
- `R_f` = risk-free rate

**Optimization Problem:**
```
minimize: w^T * Σ * w
subject to:
    w^T * μ = μ_target
    Σw_i = 1
    w_i ≥ 0
```

---

## 🔬 Optimization Methods Explained

### 1. Equal Weight (1/n)

**Approach:** Allocate equally to all assets
- **Formula:** w_i = 1/n for all assets
- **Pros:** Simple, no optimization needed
- **Cons:** Ignores correlations and individual risks
- **Use case:** Baseline comparison

### 2. Monte Carlo Simulation

**Approach:** Generate random portfolios and identify efficient ones
- **Process:** 
  1. Generate 100,000 random weight combinations
  2. Calculate return, volatility, Sharpe ratio for each
  3. Plot all portfolios
  4. Identify efficient frontier (min volatility for each return level)
- **Pros:** Visual, intuitive, finds approximate solutions
- **Cons:** Computationally expensive, approximate
- **File:** `src/optimization/monte_carlo.py`

### 3. SciPy Optimization (SLSQP)

**Approach:** Sequential Least Squares Programming
- **Algorithm:** Constrained quadratic programming
- **Objective:** Minimize volatility for target return OR maximize Sharpe ratio
- **Constraints:** 
  - Weights sum to 1
  - No short selling (weights ≥ 0)
  - Target return constraint
- **Pros:** Exact solutions, fast
- **Cons:** Can get stuck in local optima
- **File:** `src/optimization/scipy_optimizer.py`

### 4. CVXPy Convex Optimization

**Approach:** Convex optimization framework
- **Objective:** Maximize risk-adjusted return
```
  max: w^T*μ - γ*w^T*Σ*w
```
- **Parameter γ (gamma):** Risk aversion level
  - Low γ: Aggressive (higher returns, higher risk)
  - High γ: Conservative (lower risk, lower returns)
- **Features:**
  - Leverage constraints
  - Multiple risk aversion levels
  - Guaranteed global optimum (convex problem)
- **File:** `src/optimization/cvxpy_optimizer.py`

---

## 📊 Sample Output

### Console Output
```
================================================================================
SUMMARY: COMPARISON OF ALL METHODS
================================================================================
                      Method  Return (%)  Volatility (%)  Sharpe Ratio
        Equal Weight (1/n)       95.24           42.15          2.26
Monte Carlo - Max Sharpe        96.82           42.87          2.26
  Monte Carlo - Min Vol         52.89           26.85          1.97
   SciPy - Max Sharpe          100.75           44.48          2.27
     SciPy - Min Vol            52.89           26.85          1.97
================================================================================

Key Findings:
  Best Sharpe Ratio: 2.27 (SciPy Optimization - Maximum Sharpe Ratio)
  Lowest Volatility: 26.85% (Monte Carlo - Minimum Volatility)
  Highest Return:    100.75% (SciPy Optimization - Maximum Sharpe Ratio)
```

### Generated Visualizations

1. **Price History** - Stock prices 2019-2020
2. **Returns Distribution** - Histogram of daily returns
3. **Correlation Matrix** - Asset correlations heatmap
4. **Daily Returns** - Time series of returns
5. **Efficient Frontier (MC)** - Monte Carlo portfolios
6. **Optimal Portfolios** - Max Sharpe & Min Volatility highlighted
7. **Efficient Frontier (SciPy)** - Optimization-based frontier
8. **Weight Allocation** - Bar chart of portfolio weights
9. **Weights by Risk Aversion** - Stacked bar for different γ values
10. **Leverage Comparison** - Different leverage levels compared

All saved to `results/` directory in high resolution (300 DPI)

---

## 💻 Usage Examples

### Custom Stock Selection
```python
from src.data_loader import download_stock_data, calculate_returns, annualize_statistics
from src.optimization.monte_carlo import monte_carlo_simulation

# Your custom stocks
STOCKS = ['AMZN', 'NFLX', 'NVDA', 'META']

# Download data
prices = download_stock_data(STOCKS, '2022-01-01', '2024-01-01')
returns = calculate_returns(prices)
avg_returns, cov_matrix = annualize_statistics(returns)

# Run Monte Carlo
results, weights = monte_carlo_simulation(avg_returns, cov_matrix)

print(f"Generated {len(results)} portfolios")
print(f"Max Sharpe Ratio: {results['sharpe_ratio'].max():.2f}")
```

### Find Efficient Frontier
```python
from src.optimization.scipy_optimizer import get_efficient_frontier_scipy
import numpy as np

# Define return range
returns_range = np.linspace(0.05, 0.50, 100)

# Calculate efficient frontier
efficient_portfolios = get_efficient_frontier_scipy(
    avg_returns.values,
    cov_matrix.values,
    returns_range
)

# Extract volatilities
volatilities = [p['fun'] for p in efficient_portfolios]
```

### Optimize with Risk Aversion
```python
from src.optimization.cvxpy_optimizer import optimize_risk_adjusted_return

# Conservative investor (high risk aversion)
conservative = optimize_risk_adjusted_return(avg_returns, cov_matrix, gamma_value=5.0)

# Aggressive investor (low risk aversion)
aggressive = optimize_risk_adjusted_return(avg_returns, cov_matrix, gamma_value=0.1)

print(f"Conservative Sharpe: {conservative['sharpe_ratio']:.2f}")
print(f"Aggressive Sharpe: {aggressive['sharpe_ratio']:.2f}")
```

---

## 🎓 Learning Resources

### Understanding the Code

**Start here if you're new to portfolio optimization:**

1. Read `main.py` - Shows complete workflow
2. Explore `src/data_loader.py` - Data acquisition basics
3. Study `src/portfolio_metrics.py` - Core calculations
4. Try `src/optimization/monte_carlo.py` - Most intuitive method
5. Advanced: `src/optimization/cvxpy_optimizer.py` - Mathematical optimization

**Alternative path (linear learning):**
- Start with `original_script/One_script_version.py` for a linear, easy-to-follow implementation
- Then explore the modular version to see professional organization

### Recommended Reading

- **Original Paper:** Markowitz, H. (1952). "Portfolio Selection". Journal of Finance
- **Book:** "Modern Portfolio Theory and Investment Analysis" by Elton et al.
- **Online:** Investopedia's MPT guide
- **Code:** This repository's documentation

---

## 🛠️ Technical Details

### Dependencies

**Core:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance API

**Optimization:**
- `scipy` - Scientific computing & SLSQP optimizer
- `cvxpy` - Convex optimization

**Visualization:**
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations

**Analysis:**
- `pyfolio` - Portfolio analytics

See `requirements.txt` for exact versions.

### Tested On

- Python 3.7, 3.8, 3.9, 3.10
- macOS, Linux, Windows
- Jupyter notebooks supported

---

## 📈 Key Findings (2019-2020 Data)

**Assets Analyzed:**
- **AAPL (Apple):** Stable tech giant, low volatility
- **TSLA (Tesla):** High growth, high volatility
- **DAL (Delta Airlines):** COVID-19 impact, traditional industry
- **PFE (Pfizer):** Pharmaceutical, vaccine development

**Results:**
- **Best Strategy:** SciPy Maximum Sharpe (100.75% return, 2.27 Sharpe)
- **Lowest Risk:** Minimum Volatility portfolios (26.85% volatility)
- **Diversification Benefit:** Equal weight outperforms single assets
- **Correlation Insight:** AAPL-TSLA highest positive correlation (tech sector)

---

**Contributions welcome!**

---

## 📝 License

MIT License - feel free to use for learning, research, or commercial projects.

---

## 👤 Author

**Thanos Paidoulias**
- Portfolio: [https://github.com/ThanosPaidoulias]
- LinkedIn: [https://www.linkedin.com/in/thanos-paidoulias/]

---

## 🙏 Acknowledgments

- Harry Markowitz for Modern Portfolio Theory
- William Sharpe for the Sharpe ratio
- Python open-source community
- Yahoo Finance for free data access

---

## 📚 Citation

If you use this code in your research or projects, please cite:
```bibtex
@software{paidoulias2021portfolio,
  author = {Paidoulias, Thanos},
  title = {Modern Portfolio Theory: Python Implementation},
  year = {2021},
  url = {https://github.com/ThanosPaidoulias/modern-portfolio-theory}
}
```

---

⭐ **Star this repository if you found it useful!**
