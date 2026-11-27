from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


DEFAULT_CONFIG = ExperimentConfig()
