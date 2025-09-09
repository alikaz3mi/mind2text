"""
EEG Tokenizer for LLM processing.

Manages vocabulary and converts token sequences to numerical IDs for LLM input.
Returns and uses Pydantic entities for type safety.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import logging

from ..entities.common import TokenSequence
from ..entities.features import VocabInfo

LOGGER = logging.getLogger(__name__)
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import KBinsDiscretizer
import logging

logger = logging.getLogger(__name__)

class EEGTokenizer:
    """
    Converts continuous EEG features into discrete tokens for LLM processing.
    Features are binned and converted to symbolic representations.
    """
    
    def __init__(self,
                 n_bins: int = 3,
                 strategy: str = 'quantile',
                 feature_names: Optional[List[str]] = None):
        """
        Initialize EEG tokenizer.
        
        Args:
            n_bins: Number of discrete bins for each feature
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            feature_names: List of feature names for token generation
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.feature_names = feature_names
        self.discretizers = {}
        self.is_fitted = False
        
        # Define bin labels
        if n_bins == 3:
            self.bin_labels = ['LOW', 'MEDIUM', 'HIGH']
        elif n_bins == 5:
            self.bin_labels = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
        else:
            self.bin_labels = [f'BIN_{i}' for i in range(n_bins)]
            
        # Channel groupings for contextual tokens
        self.channel_groups = {
            'central': ['C3', 'C1', 'Cz', 'C2', 'C4'],
            'frontal': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4'],
            'parietal': ['CP3', 'CP1', 'CPz', 'CP2', 'CP4'],
            'left_motor': ['C3', 'FC3', 'CP3'],
            'right_motor': ['C4', 'FC4', 'CP4']
        }
        
    def fit(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> 'EEGTokenizer':
        """
        Fit tokenizer to training features.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        if feature_names is not None:
            self.feature_names = feature_names
            
        n_features = features.shape[1]
        
        # Create discretizer for each feature
        for feature_idx in range(n_features):
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            
            # Fit to feature column
            feature_data = features[:, feature_idx].reshape(-1, 1)
            discretizer.fit(feature_data)
            
            self.discretizers[feature_idx] = discretizer
            
        self.is_fitted = True
        logger.info(f"Fitted tokenizer with {n_features} features using {self.strategy} strategy")
        
        return self
    
    def transform(self, features: np.ndarray) -> List[List[str]]:
        """
        Transform features into token sequences.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            List of token sequences for each sample
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")
            
        n_samples, n_features = features.shape
        token_sequences = []
        
        for sample_idx in range(n_samples):
            sample_tokens = []
            
            for feature_idx in range(n_features):
                # Get feature value and discretize
                feature_value = features[sample_idx, feature_idx].reshape(-1, 1)
                bin_idx = int(self.discretizers[feature_idx].transform(feature_value)[0, 0])
                
                # Create token
                if self.feature_names:
                    feature_name = self.feature_names[feature_idx]
                    bin_label = self.bin_labels[bin_idx]
                    token = f"{feature_name}_{bin_label}"
                else:
                    token = f"FEAT_{feature_idx}_{self.bin_labels[bin_idx]}"
                    
                sample_tokens.append(token)
                
            token_sequences.append(sample_tokens)
            
        return token_sequences
    
    def fit_transform(self, features: np.ndarray, 
                     feature_names: Optional[List[str]] = None) -> List[List[str]]:
        """
        Fit tokenizer and transform features in one step.
        
        Args:
            features: Feature array
            feature_names: Optional feature names
            
        Returns:
            List of token sequences
        """
        return self.fit(features, feature_names).transform(features)
    
    def create_structured_tokens(self, features: np.ndarray) -> List[str]:
        """
        Create structured token sequences with channel grouping.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            List of structured token sequences
        """
        if not self.feature_names:
            raise ValueError("Feature names required for structured tokens")
            
        token_sequences = self.transform(features)
        structured_sequences = []
        
        for tokens in token_sequences:
            # Group tokens by channel and band
            channel_tokens = {}
            
            for token in tokens:
                # Parse token (format: band_channel_level)
                parts = token.split('_')
                if len(parts) >= 3:
                    band = parts[0]
                    channel = parts[1]
                    level = parts[2]
                    
                    if channel not in channel_tokens:
                        channel_tokens[channel] = {}
                    channel_tokens[channel][band] = level
                    
            # Create structured sequence
            structured_parts = []
            
            # Add channel group summaries
            for group_name, channels in self.channel_groups.items():
                group_tokens = []
                for channel in channels:
                    if channel in channel_tokens:
                        # Find dominant frequency band
                        band_levels = channel_tokens[channel]
                        dominant_band = self._find_dominant_band(band_levels)
                        if dominant_band:
                            group_tokens.append(f"{dominant_band}_{channel}")
                            
                if group_tokens:
                    structured_parts.append(f"[{group_name.upper()}:{' '.join(group_tokens)}]")
                    
            structured_sequence = ' '.join(structured_parts)
            structured_sequences.append(structured_sequence)
            
        return structured_sequences
    
    def _find_dominant_band(self, band_levels: Dict[str, str]) -> Optional[str]:
        """
        Find the most prominent frequency band for a channel.
        
        Args:
            band_levels: Dictionary mapping band names to levels
            
        Returns:
            Name of dominant frequency band
        """
        # Priority order for motor imagery
        band_priority = ['beta', 'alpha', 'gamma', 'theta', 'delta']
        
        # Find highest level bands
        high_bands = [band for band, level in band_levels.items() 
                     if level in ['HIGH', 'VERY_HIGH']]
        
        # Return highest priority band that is high
        for band in band_priority:
            if band in high_bands:
                return band
                
        # If no high bands, return any medium band
        medium_bands = [band for band, level in band_levels.items() 
                       if level == 'MEDIUM']
        
        if medium_bands:
            return medium_bands[0]
            
        return None
    
    def create_contextual_tokens(self, features: np.ndarray, 
                               labels: Optional[np.ndarray] = None) -> List[str]:
        """
        Create contextual token sequences with task-relevant information.
        
        Args:
            features: Feature array
            labels: Optional class labels for context
            
        Returns:
            List of contextual token sequences
        """
        structured_tokens = self.create_structured_tokens(features)
        contextual_sequences = []
        
        # Motor imagery context mapping
        task_context = {
            1: "TASK_LEFT_HAND",
            2: "TASK_RIGHT_HAND", 
            3: "TASK_FEET",
            4: "TASK_TONGUE"
        }
        
        for i, tokens in enumerate(structured_tokens):
            contextual_parts = []
            
            # Add task context if labels available
            if labels is not None and i < len(labels):
                task_token = task_context.get(labels[i], "TASK_UNKNOWN")
                contextual_parts.append(f"<{task_token}>")
                
            # Add EEG pattern tokens
            contextual_parts.append(tokens)
            
            # Add pattern summary
            summary = self._create_pattern_summary(tokens)
            if summary:
                contextual_parts.append(f"<PATTERN:{summary}>")
                
            contextual_sequence = ' '.join(contextual_parts)
            contextual_sequences.append(contextual_sequence)
            
        return contextual_sequences
    
    def _create_pattern_summary(self, tokens: str) -> str:
        """
        Create a high-level summary of EEG patterns.
        
        Args:
            tokens: Structured token sequence
            
        Returns:
            Pattern summary string
        """
        # Analyze tokens for common motor imagery patterns
        summary_parts = []
        
        # Check for lateralization (left vs right motor cortex activity)
        if 'LEFT_MOTOR' in tokens and 'RIGHT_MOTOR' in tokens:
            if 'beta' in tokens or 'alpha' in tokens:
                if tokens.count('LEFT_MOTOR') > tokens.count('RIGHT_MOTOR'):
                    summary_parts.append("LEFT_DOMINANT")
                elif tokens.count('RIGHT_MOTOR') > tokens.count('LEFT_MOTOR'):
                    summary_parts.append("RIGHT_DOMINANT")
                else:
                    summary_parts.append("BILATERAL")
                    
        # Check for frequency characteristics
        if 'beta' in tokens:
            summary_parts.append("BETA_ACTIVE")
        if 'alpha' in tokens:
            summary_parts.append("ALPHA_PRESENT")
            
        return '_'.join(summary_parts) if summary_parts else "GENERAL"
    
    def get_vocabulary(self) -> List[str]:
        """
        Get the complete vocabulary of possible tokens.
        
        Returns:
            List of all possible tokens
        """
        vocabulary = set()
        
        if self.feature_names:
            for feature_name in self.feature_names:
                for bin_label in self.bin_labels:
                    vocabulary.add(f"{feature_name}_{bin_label}")
        else:
            # Generic feature tokens
            for i in range(len(self.discretizers)):
                for bin_label in self.bin_labels:
                    vocabulary.add(f"FEAT_{i}_{bin_label}")
                    
        # Add special tokens
        special_tokens = [
            '<TASK_LEFT_HAND>', '<TASK_RIGHT_HAND>', '<TASK_FEET>', '<TASK_TONGUE>',
            '[CENTRAL:', '[FRONTAL:', '[PARIETAL:', '[LEFT_MOTOR:', '[RIGHT_MOTOR:',
            '<PATTERN:', 'LEFT_DOMINANT', 'RIGHT_DOMINANT', 'BILATERAL',
            'BETA_ACTIVE', 'ALPHA_PRESENT', 'GENERAL'
        ]
        
        vocabulary.update(special_tokens)
        
        return sorted(list(vocabulary))
    
    def save_tokenizer(self, filepath: str) -> None:
        """
        Save fitted tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        import pickle
        
        tokenizer_data = {
            'discretizers': self.discretizers,
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'feature_names': self.feature_names,
            'bin_labels': self.bin_labels,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
            
        logger.info(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load_tokenizer(cls, filepath: str) -> 'EEGTokenizer':
        """
        Load fitted tokenizer from file.
        
        Args:
            filepath: Path to load tokenizer from
            
        Returns:
            Loaded tokenizer instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
            
        tokenizer = cls(
            n_bins=tokenizer_data['n_bins'],
            strategy=tokenizer_data['strategy'],
            feature_names=tokenizer_data['feature_names']
        )
        
        tokenizer.discretizers = tokenizer_data['discretizers']
        tokenizer.bin_labels = tokenizer_data['bin_labels']
        tokenizer.is_fitted = tokenizer_data['is_fitted']
        
        logger.info(f"Tokenizer loaded from {filepath}")
        
        return tokenizer
