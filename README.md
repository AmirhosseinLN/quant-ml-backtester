# Quant ML Backtester  
Machine Learning–Driven Trading Strategy with Backtesting & Interactive Dashboard

---

## Overview

This project implements a full quantitative research pipeline for developing and evaluating systematic trading strategies using both traditional approaches and machine learning models.

The primary objective is to compare a classical momentum-based strategy with ML-driven signal generation and assess their performance under realistic backtesting conditions.

---

## Project Structure
quant-ml-backtester/
│
├── data/
│   ├── raw/        # Raw downloaded data
│   ├── interim/    # Intermediate cleaned datasets
│   └── processed/  # Final modeling dataset
│
├── notebooks/      # Research, experimentation, and prototyping
├── src/quant_ml/   # Core modules (data, features, models, backtesting)
├── artifacts/      # Stored outputs (metrics, curves, feature importance)
├── dashboard/      # Streamlit application
│
├── requirements.txt
└── README.md

---

## Data Sources

Market and macroeconomic data are sourced from:

- **Yahoo Finance**  
  Exchange-traded funds used as the trading universe:
  - SPY (S&P 500)
  - QQQ (Nasdaq 100)
  - IWM (Russell 2000)
  - EEM (Emerging Markets)
  - GLD (Gold)
  - TLT (Treasuries)

- **FRED (Federal Reserve Economic Data)**  
  Macroeconomic indicators:
  - Interest rates
  - Inflation (CPI)
  - Unemployment rate

---

## Portfolio Construction

The strategy operates as a systematic cross-sectional allocation model:

- Long-only portfolio
- Weekly rebalancing frequency
- Top-N asset selection based on signal ranking
- Equal-weight allocation across selected assets

Example:
If Top-N = 3 → each asset receives approximately 33% allocation

---

## Signal Generation

Two signal frameworks are implemented:

- **Momentum (Baseline)**  
  Based on historical returns and relative strength ranking

- **Machine Learning Models**  
  Predict probability of positive forward returns using engineered features

---

## Feature Engineering

The feature set includes:

- Moving averages and price ratios
- Volatility measures (rolling standard deviation)
- Volume-based signals (z-score, ratios)
- Macroeconomic indicators (rates, inflation, unemployment)

---

## Strategies Implemented

### Momentum Baseline
A rules-based allocation model selecting assets with the strongest recent performance.

### Logistic Regression (Linear Model)
A probabilistic classifier estimating the likelihood of positive forward returns.  
Emphasizes interpretability and stability.

### Random Forest (Ensemble Model)
A nonlinear model capturing interactions and regime-dependent relationships in the data.

---

## Backtesting Framework

- Out-of-sample evaluation
- Weekly portfolio rebalancing
- Transaction cost assumption: 0.1% per trade
- Portfolio returns computed via equal-weight aggregation
- Performance evaluated using standard risk metrics

---

## Results (Out-of-Sample)

| Strategy              | Return | Volatility | Sharpe | Max Drawdown |
|---------------------|--------|-----------|--------|--------------|
| Momentum Baseline    | 0.19%  | 15.26%    | 0.01   | -39.5%       |
| Logistic Regression  | 9.29%  | 16.19%    | 0.55   | -20.3%       |
| Random Forest        | 10.45% | 16.29%    | 0.61   | -26.9%       |

Machine learning strategies significantly outperform the classical momentum benchmark in both absolute and risk-adjusted terms.

---

## Dashboard

An interactive Streamlit dashboard is included for exploration and analysis:

- Cumulative performance (equity curves)
- Drawdown analysis
- Strategy comparison metrics
- Feature importance visualization

### Run locally

```bash
streamlit run dashboard/app.py


### Future Work
- Walk-forward validation
- Gradient boosting models (XGBoost / LightGBM)
- Regime detection and dynamic allocation
- Online deployment of the dashboard

## Author

**Amirhossein Latifinavid**  
[LinkedIn Profile](https://www.linkedin.com/in/amirhossein-latifinavid-5923272a7)