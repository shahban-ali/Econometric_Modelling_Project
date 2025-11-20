"""
Sprint 2 – Weekly Regime Classifier (Reference Implementation)

Author: Shahban
Role: Econometrics / Priors Design

This module implements a simple, deterministic weekly regime classifier
based on:

- VIX z-score
- Cross-asset correlation z-score
- Realized volatility z-score

It uses a composite score and probabilistic mapping with hysteresis
and confirmation ticks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class RegimeState:
    """Internal state of the regime classifier."""
    current_regime: str = "normal"
    previous_regime: str = "normal"
    regime_timestamp: Optional[pd.Timestamp] = None
    enter_counter: int = 0
    exit_counter: int = 0


class RegimeClassifier:
    """
    Regime classifier driven by a threshold configuration JSON.

    Expected feature columns in the input DataFrame:
      - vix_z
      - corr_z
      - rv_z (optional, can be missing if disabled)
    """

    def __init__(self, thresholds_path: str = "regime_thresholds.json"):
        self.thresholds_path = thresholds_path
        self.config = self._load_thresholds(thresholds_path)
        self.state = RegimeState()

        pm = self.config.get("probability_mapping", {})
        self.a = float(pm.get("a", 1.0))
        self.b = float(pm.get("b", 0.0))
        self.clamp_min = float(pm.get("clamp_min", 0.01))
        self.clamp_max = float(pm.get("clamp_max", 0.99))

        hyst = self.config.get("hysteresis", {})
        self.prob_enter = float(hyst.get("prob_enter", 0.60))
        self.prob_exit = float(hyst.get("prob_exit", 0.40))
        self.confirm_ticks = int(hyst.get("confirm_ticks", 2))

        fb = self.config.get("fallback", {})
        self.fallback_regime = fb.get("default_regime", "normal")
        self.fallback_prob = float(fb.get("default_probability", 0.0))

    @staticmethod
    def _load_thresholds(path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Thresholds file not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def reset(self) -> None:
        """Reset internal state to default."""
        self.state = RegimeState()

    def _sigmoid(self, x: float) -> float:
        """Standard logistic sigmoid with clamping."""
        val = 1.0 / (1.0 + float(np.exp(-(self.a * x + self.b))))
        return float(np.clip(val, self.clamp_min, self.clamp_max))

    def _compute_probability(self, row: pd.Series) -> Optional[float]:
        """
        Compute the probability of high_vol regime from z-scores.
        Returns None if features are missing.
        """
        # Try to read z-scores; if not available, return None
        try:
            vix_z = float(row.get("vix_z", np.nan))
            corr_z = float(row.get("corr_z", np.nan))
        except Exception:
            return None

        values = [vix_z, corr_z]

        # Optional realized vol
        if "rv_z" in row.index:
            try:
                values.append(float(row["rv_z"]))
            except Exception:
                pass

        # Filter out NaNs
        values = [v for v in values if np.isfinite(v)]
        if not values:
            return None

        z_max = max(values)
        return self._sigmoid(z_max)

    def _apply_hysteresis(self, prob: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Update internal state based on new probability and return result.
        """
        s = self.state
        prev_regime = s.current_regime

        # ENTER high_vol
        if s.current_regime == "normal":
            if prob >= self.prob_enter:
                s.enter_counter += 1
                s.exit_counter = 0
                if s.enter_counter >= self.confirm_ticks:
                    s.previous_regime = s.current_regime
                    s.current_regime = "high_vol"
                    s.regime_timestamp = timestamp
                    s.enter_counter = 0
            else:
                s.enter_counter = 0

        # EXIT to normal
        elif s.current_regime == "high_vol":
            if prob <= self.prob_exit:
                s.exit_counter += 1
                s.enter_counter = 0
                if s.exit_counter >= self.confirm_ticks:
                    s.previous_regime = s.current_regime
                    s.current_regime = "normal"
                    s.regime_timestamp = timestamp
                    s.exit_counter = 0
            else:
                s.exit_counter = 0

        # Package result
        result = {
            "regime": s.current_regime,
            "probability": float(prob),
            "previous_regime": prev_regime,
            "regime_timestamp": None if s.regime_timestamp is None else s.regime_timestamp.isoformat()
        }
        return result

    def classify_row(self, row: pd.Series, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Classify a single row of features.

        Parameters:
            row: pd.Series with z-scores (vix_z, corr_z, optional rv_z)
            timestamp: optional timestamp (if None, use row.name if available)

        Returns:
            dict with regime, probability, previous_regime, regime_timestamp
        """
        if timestamp is None:
            # If row comes from a DataFrame, use index as timestamp
            if isinstance(row.name, (pd.Timestamp, pd.Period)):
                timestamp = pd.to_datetime(row.name)
            else:
                timestamp = pd.Timestamp.utcnow()

        prob = self._compute_probability(row)
        if prob is None:
            # Fallback: missing or invalid data
            self.state.previous_regime = self.state.current_regime
            self.state.current_regime = self.fallback_regime
            self.state.regime_timestamp = timestamp
            self.state.enter_counter = 0
            self.state.exit_counter = 0
            return {
                "regime": self.fallback_regime,
                "probability": self.fallback_prob,
                "previous_regime": self.state.previous_regime,
                "regime_timestamp": timestamp.isoformat()
            }

        return self._apply_hysteresis(prob, timestamp)

    def classify_series(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Classify an entire DataFrame of features, row by row.

        Parameters:
            features: DataFrame with columns vix_z, corr_z, optional rv_z

        Returns:
            DataFrame with columns: regime, probability, previous_regime, regime_timestamp
        """
        self.reset()
        records = []

        for idx, row in features.iterrows():
            result = self.classify_row(row, timestamp=pd.to_datetime(idx))
            records.append(result)

        out = pd.DataFrame(records, index=features.index)
        return out
