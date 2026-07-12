"""Single source of truth for the polynomial_kinematic_ml study. No tunables elsewhere."""
from dataclasses import dataclass, field


@dataclass
class StudyConfig:
    asset: str = "btcusdt"
    start_date: str = "2024-01-01"
    stop_date: str = "2025-12-31"
    lookback_windows: list[int] = field(default_factory=lambda: [10, 20, 30, 60, 120, 240])
    forward_horizons: list[int] = field(default_factory=lambda: [60, 120, 180, 240])
    bps_multiplier: float = 0.0030
    cv_splits: int = 5
    dataset_output_path: str = "data/processed/kinematic_study_dataset.parquet"
    oof_output_path: str = "data/processed/kinematic_oof_predictions.parquet"
    metrics_output_path: str = "scripts/studies/polynomial_kinematic_ml/metrics.json"
    classification_threshold: float = 0.5
    random_state: int = 42
    lgbm_params: dict = field(default_factory=lambda: {
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "verbosity": -1,
    })
    catboost_params: dict = field(default_factory=lambda: {
        "depth": 4,
        "learning_rate": 0.03,
        "iterations": 400,
        "verbose": 0,
        "allow_writing_files": False,
    })
    elasticnet_params: dict = field(default_factory=lambda: {
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratio": 0.5,
        "C": 1.0,
        "max_iter": 200,
    })


CONFIG = StudyConfig()
