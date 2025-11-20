
Feature Specification - Regime Detection (Sprint 2)
Author: Shahban
Date: 2025-01-01

-------------------------------------------------------------
1. Objective
-------------------------------------------------------------
Define the features, thresholds, and decision process used to
classify the market regime as:
- normal
- high_vol

The output is consumed by:
- covariance engine
- priors API
- monitoring and evidence systems

-------------------------------------------------------------
2. Inputs
-------------------------------------------------------------

2.1 Market Returns (Y)
Shape: T x N
N = 300 stable S&P500 tickers
Loaded through safe_load_returns().

2.2 Factors (F)
Includes:
- Fama-French 5 factors
- Momentum factor
- Sector ETFs optional extension

2.3 VIX
Daily VIX from Yahoo Finance
Converted to monthly averages.

-------------------------------------------------------------
3. Engineered Features
-------------------------------------------------------------

3.1 VIX 30-Day Average
vix_30d[t] = mean(VIX[t-29 to t])

3.2 Cross-Asset Correlation
1. Compute correlation matrix from Y monthly returns
2. Extract upper triangle
3. Use median correlation

3.3 Z-Score Normalization
z = (value - rolling_mean) / rolling_std

-------------------------------------------------------------
4. Threshold Summary
-------------------------------------------------------------

ENTER high_vol if any condition holds:
- vix_z >= 1.2
- corr_z >= 1.5
- vix_abs >= 22
- corr_abs >= 0.70
- probability >= 0.60

EXIT to normal if:
- vix_z <= 0.5
- corr_z <= 0.7
- probability <= 0.40

-------------------------------------------------------------
5. Probability Mapping
-------------------------------------------------------------

p = 1 / (1 + exp(-(a * zmax + b)))

Where:
- zmax = max(vix_z, corr_z)
- a = 1.0
- b = 0.0
- probability clamped between 0.01 and 0.99

-------------------------------------------------------------
6. Final Regime Decision
-------------------------------------------------------------

If p >= 0.60 for 2 consecutive points -> high_vol
If p <= 0.40 for 2 consecutive points -> normal
Otherwise stay in previous regime.

-------------------------------------------------------------
7. Output Format
-------------------------------------------------------------

{
  "regime": "high_vol",
  "probability": 0.72,
  "timestamp": "2025-01-01"
}

-------------------------------------------------------------
8. Evidence Required
-------------------------------------------------------------

- backtest_summary.json
- current_sample.json
- latency_samples.json

End of Document
