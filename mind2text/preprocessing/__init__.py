"""
EEG Preprocessing Pipeline for Cognitive State Classification

This module provides comprehensive EEG preprocessing capabilities including:
- Raw EEG data loading and filtering (ds004148 BIDS format)
- Feature extraction (band power, spectral features)
- Signal segmentation and trial extraction
- Data cleaning and artifact removal
- Returns Pydantic entities for type safety and validation
"""

from .eeg_loader import EEGDataLoader
from .feature_extractor import FeatureExtractor
from .signal_processor import SignalProcessor
from .trial_segmenter import TrialSegmenter

__all__ = [
    'EEGDataLoader',
    'FeatureExtractor', 
    'SignalProcessor',
    'TrialSegmenter'
]