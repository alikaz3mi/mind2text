"""
Models module for EEG-to-LLM classification.

This module provides:
- LLM-based models with LoRA fine-tuning
- Baseline classification models (CNN, SVM)
- Training and inference utilities
- Returns Pydantic entities for type safety
"""

from .llm_classifier import LLMClassifier
from .baseline_models import CNNBaseline, SVMBaseline
from .trainer import EEGTrainer

__all__ = [
    'LLMClassifier',
    'CNNBaseline',
    'SVMBaseline',
    'EEGTrainer'
]