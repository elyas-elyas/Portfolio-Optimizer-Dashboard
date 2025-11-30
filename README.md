# Portfolio Optimizer: MPT & Machine Learning Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://imgshields.io/badge/Status-Feature--Rich-success)

An investment dashboard for portfolio optimization using Modern Portfolio Theory (Markowitz), enhanced with optional **Machine Learning (ARMA/GARCH)** for dynamic parameter forecasting and **Monte Carlo Simulations** for risk forecasting.

---

## Project Objective

Provide financial analysts and investors with a robust, interactive tool to optimize asset allocation by choosing between two analytical modes:

1.  **Classic MPT**: Uses historical averages to define the Efficient Frontier, coupled with Monte Carlo forecasting for long-term risk projection.
2.  **ML Forecast (ARMA/GARCH)**: Uses predictive models (GARCH for volatility) to inform optimization, offering an allocation based on projected future market conditions.

---

## Key Features

| Feature | Description | Analysis Section |
|-------------------------------------|--------------------------------------------------------------------------------|--------------------------|
| **Optimization Basis Selection** | Choose between Historical Averages (Classic MPT) or ML Predictions (ARMA/GARCH) for parameters. | Configuration |
| **Monte Carlo Simulation** | Forecast portfolio value and risk (P10, P50, P90 percentiles) over a defined horizon (only in Classic MPT mode). | Forecasting |
| **Automatic Rebalancing** | Backtest the optimal portfolio using specified periodic rebalancing (Monthly, Quarterly, Annually). | Backtesting |
| **Risk-Adjusted Sharpe** | Calculate the Sharpe Ratio using a configurable Risk-Free Rate. | All Sections |
| **Benchmark Comparison** | Compare optimized portfolio performance against major indices (e.g., S&P 500). | Backtesting |
| **Custom Tickers & Lists** | Flexible asset selection via custom input or predefined categorized lists. | Sidebar |

---

## Methodology

### 1. Optimization Core
- **Model**: Modern Portfolio Theory (Markowitz).
- **Objectives**: Max Sharpe Ratio Portfolio, Min Volatility Portfolio.
- **Risk Measure**: Annualized Volatility.

### 2. Machine Learning Forecast (ML Mode)
- **Model Used**: AR(1)-GARCH(1,1).
- **Goal**: Predict the next period's mean return and volatility for each asset.
- **Implementation Note**: Historical correlation is maintained, but volatility components are scaled using the GARCH forecast for robust covariance estimation.

### 3. Monte Carlo Simulation (Classic Mode Only)
- **Method**: Geometric Brownian Motion (GBM).
- **Inputs**: Portfolio Mean Return ($\mu$) and Volatility ($\sigma$) calculated from historical data.
- **Output**: Value distribution (P10, P50, P90) over $N$ simulated years.

### 4. Backtesting
- **Strategy**: Rebalance to initial optimal weights at chosen frequency.
- **Metrics**: Total/Annualized Return, Sharpe Ratio (using RFR), Max Drawdown.

---

## Technologies Used

- **Python 3.8+**
- `streamlit` - Frontend development
- `yfinance` - Financial data download
- `pandas` & `numpy` - Data manipulation
- `plotly` - Interactive visualizations
- `scipy` - Optimization solver (SLSQP)
- `arch` - GARCH modeling (for ML Forecast mode)

---

## Project Structure
```
portfolio-optimizer-ml/
│
├── app.py                     # Main Streamlit application script
└── requirements.txt           # Python dependencies
```

---

## Installation and Usage

### Prerequisites
```bash
Python 3.8 or higher
pip
```

### Installation

1.  **Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)[your-username]/portfolio-optimizer-ml.git
cd portfolio-optimizer-ml
```

2.  **Install dependencies**
    *Note: The `arch` library is required for the ML mode.*
```bash
pip install -r requirements.txt
```

3.  **Launch the Streamlit app**
```bash
streamlit run app.py
```

The application will open automatically in your web browser, typically at `http://localhost:8501`.

---

## Deployment Status

This application is designed for easy deployment on **Streamlit Community Cloud**.

| Deployment Status | Link |
|-------------------|------|
| **Live App** | [] |

---

## Strengths

- **Hybrid Analysis**: Offers both robust historical MPT and forward-looking ML-informed optimization.
- **Dynamic Risk Adjustment**: Utilizes the Risk-Free Rate in Sharpe calculation and GARCH in ML mode for volatility prediction.
- **Forecasting**: Monte Carlo simulations provide critical forward-looking probabilistic risk measures.
- **Clear Separation of Concerns**: Optimization metrics (Input) are separated from Backtesting metrics (Historical Performance).

## Limitations

- **ML Sensitivity**: The ARMA/GARCH model can be sensitive to outliers, potentially leading to unrealistic parameter forecasts (e.g., extremely high predicted volatility). **Use with caution.**
- **Transaction Costs**: Transaction costs for rebalancing and execution are not included in the backtesting metrics.
- **Execution**: The ML prediction step (`arch` library) can be slow depending on the number of tickers and historical data length.

---

## Key Concepts

| Concept | Explanation |
|-------------------------|----------------------------------------------------------|
| **MPT (Markowitz)** | Portfolio selection based on maximizing return for a given level of risk. |
| **Efficient Frontier** | The set of optimal portfolios offering the highest expected return for a specific level of risk. |
| **Sharpe Ratio** | Measures risk-adjusted return relative to a Risk-Free Rate. |
| **Rebalancing** | Process of returning the portfolio weights to the target allocation periodically. |
| **ARMA/GARCH** | Machine Learning models used to forecast the mean (ARMA) and volatility (GARCH) of time series data. |
| **Monte Carlo (GBM)** | Simulation technique using historical parameters to forecast the probabilistic distribution of future portfolio values. |

---
