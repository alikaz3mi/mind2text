"""
Algorithm module for EEG-to-LLM processing

This module provides core algorithms for:
- Feature discretization and binning
- Symbolic tokenization of EEG features
- Vocabulary management
- Token sequence generation
"""

from .symbolic_encoder import SymbolicEncoder
from .tokenizer import EEGTokenizer
from .binning import FeatureBinner

__all__ = [
    'SymbolicEncoder',
    'EEGTokenizer',
    'FeatureBinner'
]

from .tokenizer import EEGTokenizer
from .symbolic_encoder import SymbolicEncoder
from .sequence_generator import SequenceGenerator

__all__ = [
    'EEGTokenizer',
    'SymbolicEncoder', 
    'SequenceGenerator'
]