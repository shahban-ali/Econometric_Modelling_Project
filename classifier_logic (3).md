# Sprint 2 – Weekly Regime Classifier Logic

Author: Shahban
Role: Econometrics / Priors Design
File: classifier_logic.md

---

## 1. Objective

This document explains **how the weekly regime classifier works** and
how to interpret its output.

The classifier:

- Distinguishes between two regimes:
  - `normal`
  - `high_vol` (elevated stress)
- Produces a **probability p in [0, 1]** that the regime is `high_vol`.
- Uses:
  - VIX-based volatility
  - Cross-asset correlation
  - Realized volatility
- Applies **hysteresis** and **two-tick confirmation** to avoid flip–flop.
- Fails **closed**: if inputs are missing, we fall back to `normal` and raise an alert.

---

## 2. Time Scale and Data

- Evaluation frequency: **weekly**
- Features are computed on **4-week rolling windows**.
- Inputs come from:
  - Market returns (Y): shape `(T, N)`
  - Factor returns (F): shape `(T, K)`
  - VIX daily prices aggregated to weekly

The classifier itself only needs the following columns at each weekly timestamp:

- `vix_level`  – 4-week average VIX
- `corr_4w`    – 4-week median cross-asset correlation
- `rv_4w`      – 4-week realized volatility (if provided)

These features are provided either by the data pipeline or pre-computed upstream.

---

## 3. Feature Definitions (Informal)

### 3.1 VIX Feature

- `vix_level(t)`: average of daily VIX levels over the last 4 weeks.
- We transform `vix_level` into a standardized score `vix_z` using a rolling mean
  and standard deviation over approximately one year of weekly history.

### 3.2 Cross-Asset Correlation

- Universe: a subset of liquid assets (for example, the 300 assets loaded in the
  econometric loader).
- At each week:
  - Take the last 4 weekly returns.
  - Compute the correlation matrix.
  - Extract all pairwise correlations.
  - Take the **median** of those pairwise correlations.
- This yields a single number `corr_4w` between -1 and 1.
- We transform `corr_4w` into a standardized score `corr_z` using rolling statistics.

### 3.3 Realized Volatility

- We compute 4-week realized volatility `rv_4w` for a chosen index or portfolio.
- Volatility is annualized with factor 52 (weeks per year).
- `rv_4w` is converted to a z-score `rv_z`.

---

## 4. Z-Score Normalization

For each feature X, we compute a rolling z-score:

- z = (X - rolling_mean(X)) / rolling_std(X)

We ensure:

- Sufficient history before using a z-score (for example, at least 52 weeks).
- A small epsilon is applied if the rolling standard deviation is near zero.

The current implementation uses:

- `vix_z`
- `corr_z`
- `rv_z` (if enabled)

---

## 5. Composite Stress Score and Probability

### 5.1 Composite z-score

We define:

- `z_max = max(vix_z, corr_z, rv_z_if_enabled)`

This captures the **strongest stress signal** among the three features.

### 5.2 Probability Mapping

We map `z_max` to a probability p that the regime is `high_vol` using a sigmoid:

- p = 1 / (1 + exp( - (a * z_max + b) ))

Where:

- `a` controls the slope.
- `b` controls the midpoint.

In the current configuration:

- a = 1.0
- b = 0.0

We then clamp p to [0.01, 0.99] to avoid exact zeros or ones.

---

## 6. Regime Decision Logic (Without Hysteresis)

If we ignore hysteresis for a moment, the logic is:

- Evaluate features at week t.
- Compute z-scores and then p.
- If p is high, we would lean toward `high_vol`.
- If p is low, we would lean toward `normal`.

To make the system stable, hysteresis adds memory and two thresholds.

---

## 7. Hysteresis and Two-Tick Confirmation

We use two probability thresholds:

- `prob_enter = 0.60`
- `prob_exit  = 0.40`

And a confirmation counter:

- `confirm_ticks = 2`

### 7.1 Enter High-Vol Regime

If the current regime is `normal`:

1. If `p >= prob_enter`:
   - Increase an internal "enter_high_vol" counter.
   - If this condition holds for `confirm_ticks` consecutive weeks,
     switch regime to `high_vol` and reset the counters.

2. If `p < prob_enter`:
   - Reset the "enter_high_vol" counter.

### 7.2 Exit High-Vol Regime

If the current regime is `high_vol`:

1. If `p <= prob_exit`:
   - Increase an internal "exit_to_normal" counter.
   - If this holds for `confirm_ticks` consecutive weeks,
     switch regime back to `normal`.

2. If `p > prob_exit`:
   - Reset the "exit_to_normal" counter.

This design:

- Makes entering `high_vol` require **strong evidence**.
- Makes exiting `high_vol` require **sustained normalization**.

---

## 8. Fallback Behavior

If any critical input is missing or corrupted:

- We set:
  - `regime = "normal"`
  - `probability = 0.0`
- We raise a P1 alert indicating a data outage.
- This behaviour is controlled by the `fallback` block in `regime_thresholds.json`.

The idea: **if we are blind, we behave conservatively**, not as if volatility is
definitely high.

---

## 9. Determinism and Evidence

The classifier is deterministic:

- Same feature history and same configuration
  → same regime path and probabilities.

For backtesting and evidence:

- We record:
  - thresholds
  - seeds (if any random parts exist upstream)
  - data hashes
  - confusion matrix metrics

These are stored in the sprint evidence pack (see main sprint document).

---

## 10. Integration Contract

The `regime_classifier.py` reference implementation exposes a class:

- `RegimeClassifier`

It:

- Consumes weekly feature rows (for example, from a DataFrame).
- Maintains internal state:
  - current regime
  - confirmation counters
  - previous regime
  - last transition timestamp
- Returns, for each timestamp:

  - `regime`           – "normal" or "high_vol"
  - `probability`      – p in [0, 1]
  - `previous_regime`  – regime before the last transition
  - `regime_timestamp` – timestamp of the current regime assignment

This is the contract that the API and downstream components (covariance engine,
factor health, optimizer) rely on.

---
