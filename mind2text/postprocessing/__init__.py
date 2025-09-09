"""
Postprocessing module for Mind2Text.

This module provides:
- Probability calibration for improved confidence estimates
- Report generation and evaluation metrics
- Prediction aggregation and refinement
- Uses Pydantic entities for type safety
"""

from .calibrator import ProbabilityCalibrator
from .evaluator import ModelEvaluator
from .aggregator import PredictionAggregator
from .reporter import ReportGenerator

__all__ = [
    'ProbabilityCalibrator',
    'ModelEvaluator',
    'PredictionAggregator',
    'ReportGenerator'
]
