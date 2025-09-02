"""
Mind2Text: LLM-based Classification and Explainable Analysis of Motor Imagery EEG

A research framework for converting EEG motor imagery signals into symbolic representations
that can be processed by Large Language Models for both classification and explanation generation.

Author: Ali Kazemi
"""

__version__ = "0.1.0"
__author__ = "Ali Kazemi"
__email__ = "alikazemi@ieee.org"

from . import preprocessing
from . import algorithm  
from . import models

__all__ = [
    'preprocessing',
    'algorithm',
    'models'
]
