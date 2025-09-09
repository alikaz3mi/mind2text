"""
Feature binning and discretization for EEG-to-LLM processing.

Converts continuous EEG features into discrete bins for symbolic encoding.
Returns and uses Pydantic entities for type safety.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import KBinsDiscretizer
import pickle
import json
from pathlib import Path
import logging

from ..entities.common import FeatureVector
from ..entities.dataset import BinningRule

LOGGER = logging.getLogger(__name__)

class FeatureBinner:
    """
    Discretizes continuous EEG features into symbolic bins.
    Uses validated Pydantic entities for configuration and output.
    """
    
    def __init__(self, 
                 n_bins: int = 3,
                 strategy: str = 'quantile',
                 bin_labels: Optional[List[str]] = None):
        """
        Initialize feature binner.
        
        Parameters
        ----------
        n_bins : int
            Number of bins per feature
        strategy : str
            Binning strategy ('uniform', 'quantile', 'kmeans')
        bin_labels : Optional[List[str]]
            Custom labels for bins. If None, uses ['LOW', 'MEDIUM', 'HIGH'] for 3 bins
        """
        self.n_bins = n_bins
        self.strategy = strategy
        
        if bin_labels is None:
            if n_bins == 3:
                self.bin_labels = ['LOW', 'MEDIUM', 'HIGH']
            elif n_bins == 5:
                self.bin_labels = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
            else:
                self.bin_labels = [f'BIN_{i}' for i in range(n_bins)]
        else:
            self.bin_labels = bin_labels
            
        self.binner = KBinsDiscretizer(
            n_bins=n_bins, 
            encode='ordinal', 
            strategy=strategy
        )
        
        self.binning_rules: Dict[str, BinningRule] = {}
        self.is_fitted = False
        
    def fit(self, feature_vectors: List[FeatureVector]) -> None:
        """
        Fit binning rules to feature vectors.
        
        Parameters
        ----------
        feature_vectors : List[FeatureVector]
            List of feature vectors to fit binning rules
        """
        if not feature_vectors:
            raise ValueError("Cannot fit on empty feature vector list")
            
        # Collect all features
        all_features = []
        feature_names = []
        
        # Get feature structure from first vector
        first_fv = feature_vectors[0]
        n_channels = len(first_fv.channel_names)
        n_bands = len(first_fv.bands)
        
        # Create feature names
        for ch_idx, ch_name in enumerate(first_fv.channel_names):
            for band_idx, band_name in enumerate(first_fv.bands):
                feature_names.append(f"{band_name}_{ch_name}")
                
        # Extract features from all vectors
        for fv in feature_vectors:
            # Flatten feature matrix to 1D
            flat_features = []
            for ch_idx in range(len(fv.channel_names)):
                for band_idx in range(len(fv.bands)):
                    flat_features.append(fv.values[ch_idx][band_idx])
            all_features.append(flat_features)
            
        # Fit binner
        X = np.array(all_features)
        self.binner.fit(X)
        
        # Create binning rules for each feature
        bin_edges = self.binner.bin_edges_
        
        for feat_idx, feat_name in enumerate(feature_names):
            parts = feat_name.split('_')
            band_name = parts[0]
            ch_name = '_'.join(parts[1:])
            
            rule = BinningRule(
                feature_name=feat_name,
                channel_name=ch_name,
                bin_edges=bin_edges[feat_idx].tolist(),
                bin_labels=self.bin_labels.copy(),
                version="1.0"
            )
            self.binning_rules[feat_name] = rule
            
        self.is_fitted = True
        LOGGER.info(f"Fitted binner on {len(feature_vectors)} feature vectors "
                   f"with {len(feature_names)} features")
        
    def transform(self, feature_vectors: List[FeatureVector]) -> List[Dict[str, str]]:
        """
        Transform feature vectors to discrete bins.
        
        Parameters
        ----------
        feature_vectors : List[FeatureVector]
            Feature vectors to transform
            
        Returns
        -------
        List[Dict[str, str]]
            List of binned features as dictionaries mapping feature_name -> bin_label
        """
        if not self.is_fitted:
            raise ValueError("Binner must be fitted before transforming")
            
        if not feature_vectors:
            return []
            
        # Extract features
        all_features = []
        for fv in feature_vectors:
            flat_features = []
            for ch_idx in range(len(fv.channel_names)):
                for band_idx in range(len(fv.bands)):
                    flat_features.append(fv.values[ch_idx][band_idx])
            all_features.append(flat_features)
            
        # Transform to bins
        X = np.array(all_features)
        binned = self.binner.transform(X).astype(int)
        
        # Convert to labeled dictionaries
        result = []
        feature_names = list(self.binning_rules.keys())
        
        for row in binned:
            binned_dict = {}
            for feat_idx, feat_name in enumerate(feature_names):
                bin_idx = row[feat_idx]
                bin_label = self.bin_labels[bin_idx]
                binned_dict[feat_name] = bin_label
            result.append(binned_dict)
            
        return result
    
    def fit_transform(self, feature_vectors: List[FeatureVector]) -> List[Dict[str, str]]:
        """
        Fit binner and transform feature vectors in one step.
        
        Parameters
        ----------
        feature_vectors : List[FeatureVector]
            Feature vectors to fit and transform
            
        Returns
        -------
        List[Dict[str, str]]
            List of binned features
        """
        self.fit(feature_vectors)
        return self.transform(feature_vectors)
    
    def save_binning_rules(self, filepath: str) -> None:
        """
        Save binning rules to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save binning rules
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted binner")
            
        rules_dict = {}
        for name, rule in self.binning_rules.items():
            rules_dict[name] = rule.model_dump()
            
        save_data = {
            'binning_rules': rules_dict,
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'bin_labels': self.bin_labels
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        LOGGER.info(f"Saved binning rules to {filepath}")
    
    def load_binning_rules(self, filepath: str) -> None:
        """
        Load binning rules from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to load binning rules from
        """
        with open(filepath, 'r') as f:
            save_data = json.load(f)
            
        self.n_bins = save_data['n_bins']
        self.strategy = save_data['strategy']
        self.bin_labels = save_data['bin_labels']
        
        # Reconstruct binning rules
        self.binning_rules = {}
        for name, rule_dict in save_data['binning_rules'].items():
            rule = BinningRule(**rule_dict)
            self.binning_rules[name] = rule
            
        # Reconstruct binner from rules
        # This is a simplified reconstruction - in practice you might want to save the full sklearn object
        self.binner = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy=self.strategy
        )
        
        # Create dummy data to set bin edges
        feature_names = list(self.binning_rules.keys())
        dummy_X = np.random.rand(10, len(feature_names))
        self.binner.fit(dummy_X)
        
        # Set actual bin edges from rules
        bin_edges = []
        for feat_name in feature_names:
            rule = self.binning_rules[feat_name]
            bin_edges.append(np.array(rule.bin_edges))
        self.binner.bin_edges_ = bin_edges
        
        self.is_fitted = True
        LOGGER.info(f"Loaded binning rules from {filepath}")
    
    def get_binning_summary(self) -> Dict[str, Dict]:
        """
        Get summary of binning rules.
        
        Returns
        -------
        Dict[str, Dict]
            Summary of binning configuration and rules
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
            
        summary = {
            'status': 'fitted',
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'bin_labels': self.bin_labels,
            'n_features': len(self.binning_rules),
            'features': list(self.binning_rules.keys())
        }
        
        return summary
