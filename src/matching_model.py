"""Matching Model module supporting Random Forest and alternative classifiers.

Provides:
- Abstracted matching model interface
- Factory for model creation (Random Forest, LightGBM, Gradient Boosting)
- Class imbalance handling (balanced weights)
- Feature importance extraction
- Model persistence (joblib) with metadata
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Optional LightGBM support
try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


class MatchingClassifier:
    """Wrapper around scikit-learn compatible binary classifiers for entity matching."""

    def __init__(
        self,
        model_type: str = "random_forest",
        feature_names: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        self.feature_names = feature_names or []
        self.model_params = model_params or {}
        self.model = self._init_underlying_model()
        self.is_fitted = False

    def _init_underlying_model(self):
        """Initialize the underlying classifier based on model_type."""
        if self.model_type in ("rf", "random_forest", "randomforest"):
            default_params = {
                "n_estimators": 100,
                "max_depth": 14,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)

        elif self.model_type in ("lgb", "lightgbm"):
            if not _HAS_LIGHTGBM:
                raise ImportError("lightgbm is not installed. Install it or use 'random_forest'.")
            default_params = {
                "n_estimators": 150,
                "learning_rate": 0.08,
                "num_leaves": 31,
                "max_depth": 8,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            default_params.update(self.model_params)
            return lgb.LGBMClassifier(**default_params)

        elif self.model_type in ("gb", "gradient_boosting"):
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42,
            }
            default_params.update(self.model_params)
            return GradientBoostingClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Choose 'random_forest', 'lightgbm', or 'gradient_boosting'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MatchingClassifier":
        """Train the classifier on feature matrix X and binary labels y."""
        if len(y) == 0:
            raise ValueError("Training data y is empty.")
        if len(np.unique(y)) < 2:
            raise ValueError(f"Training data must contain both positive (1) and negative (0) labels. Found: {np.unique(y)}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability (class 1) for every row in X."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        
        probas = self.model.predict_proba(X)
        # Check class ordering
        classes = list(self.model.classes_)
        if 1 in classes:
            pos_idx = classes.index(1)
            return probas[:, pos_idx]
        return probas[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match labels using given probability threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dict of {feature_name: importance} sorted descending."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            names = self.feature_names if self.feature_names else [f"f_{i}" for i in range(len(importances))]
            res = dict(zip(names, [float(x) for x in importances]))
            return dict(sorted(res.items(), key=lambda kv: kv[1], reverse=True))
        return {}

    def save(self, filepath: str, threshold: Optional[float] = None) -> None:
        """Save model and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "threshold": threshold,
        }
        joblib.dump(payload, filepath)

        # Also save JSON metadata for easy inspection
        meta_filepath = os.path.splitext(filepath)[0] + "_meta.json"
        meta = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "threshold": threshold,
            "feature_importances": self.get_feature_importances(),
        }
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> Tuple["MatchingClassifier", Optional[float]]:
        """Load model and threshold from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        payload = joblib.load(filepath)
        instance = cls(
            model_type=payload.get("model_type", "random_forest"),
            feature_names=payload.get("feature_names", []),
            model_params=payload.get("model_params", {}),
        )
        instance.model = payload["model"]
        instance.is_fitted = payload.get("is_fitted", True)
        threshold = payload.get("threshold", 0.5)
        return instance, threshold
