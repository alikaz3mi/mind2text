"""
Probability calibration for improved confidence estimates.

Provides temperature scaling and isotonic regression for calibrating
model predictions. Uses Pydantic entities for type safety.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import logging

from ..entities.modeling import Prediction, PredictionCalibrated, CalibrationParams
from ..entities.reports import CalibrationMetrics

LOGGER = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    Calibrates prediction probabilities for improved confidence estimates.
    Uses validated Pydantic entities for input and output.
    """
    
    def __init__(self, method: str = 'temperature'):
        """
        Initialize probability calibrator.
        
        Parameters
        ----------
        method : str
            Calibration method ('temperature' or 'isotonic')
        """
        self.method = method
        self.calibration_params: Optional[CalibrationParams] = None
        self.is_fitted = False
        
        if method == 'temperature':
            self.calibrator = LogisticRegression()
        elif method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, 
            predictions: List[Prediction], 
            true_labels: List[str]) -> CalibrationParams:
        """
        Fit calibration parameters on validation predictions.
        
        Parameters
        ----------
        predictions : List[Prediction]
            Model predictions to calibrate
        true_labels : List[str]
            True class labels
            
        Returns
        -------
        CalibrationParams
            Fitted calibration parameters entity
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Number of predictions must match number of labels")
        
        # Extract probabilities and convert labels
        probs = np.array([pred.probs for pred in predictions])
        
        # Create label mapping
        unique_labels = sorted(set(true_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        y_true = np.array([label_to_id[label] for label in true_labels])
        
        if self.method == 'temperature':
            # Temperature scaling: fit single parameter to scale logits
            logits = np.array([pred.logits for pred in predictions])
            
            # Fit temperature parameter
            best_temp = self._find_best_temperature(logits, y_true)
            
            self.calibration_params = CalibrationParams(
                method='temperature',
                temperature=float(best_temp),
                per_class_params=None,
                version="1.0"
            )
            
        elif self.method == 'isotonic':
            # Isotonic regression: fit non-parametric calibration
            # Use maximum probability as confidence score
            max_probs = np.max(probs, axis=1)
            y_binary = (np.argmax(probs, axis=1) == y_true).astype(int)
            
            self.calibrator.fit(max_probs, y_binary)
            
            self.calibration_params = CalibrationParams(
                method='isotonic',
                temperature=None,
                per_class_params={'isotonic_fitted': True},
                version="1.0"
            )
        
        self.is_fitted = True
        LOGGER.info(f"Fitted {self.method} calibration on {len(predictions)} samples")
        
        return self.calibration_params
    
    def _find_best_temperature(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Find optimal temperature parameter using line search."""
        from scipy.optimize import minimize_scalar
        
        def temperature_loss(temp: float) -> float:
            scaled_logits = logits / temp
            probs = self._softmax(scaled_logits)
            return self._cross_entropy_loss(probs, y_true)
        
        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        n_samples = len(y_true)
        log_probs = np.log(probs + 1e-15)  # Add small epsilon for numerical stability
        return -np.sum(log_probs[np.arange(n_samples), y_true]) / n_samples
    
    def transform(self, predictions: List[Prediction]) -> List[PredictionCalibrated]:
        """
        Apply calibration to predictions.
        
        Parameters
        ----------
        predictions : List[Prediction]
            Predictions to calibrate
            
        Returns
        -------
        List[PredictionCalibrated]
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        calibrated_predictions = []
        
        for pred in predictions:
            if self.method == 'temperature':
                # Apply temperature scaling
                temp = self.calibration_params.temperature
                scaled_logits = np.array(pred.logits) / temp
                calibrated_probs = self._softmax(scaled_logits.reshape(1, -1))[0]
                
            elif self.method == 'isotonic':
                # Apply isotonic regression
                max_prob = max(pred.probs)
                calibrated_max_prob = self.calibrator.predict([max_prob])[0]
                
                # Scale all probabilities to maintain distribution shape
                scale_factor = calibrated_max_prob / max_prob if max_prob > 0 else 1.0
                calibrated_probs = np.array(pred.probs) * scale_factor
                
                # Renormalize to ensure sum to 1
                calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
            
            # Get calibrated prediction class
            calibrated_class_idx = np.argmax(calibrated_probs)
            class_names = ["memory", "mathematic", "music", "eyesopen", "eyesclosed"]
            calibrated_class = class_names[calibrated_class_idx]
            
            # Check for abstention (low confidence threshold)
            max_calibrated_prob = np.max(calibrated_probs)
            abstained = max_calibrated_prob < 0.5  # Configurable threshold
            
            calibrated_pred = PredictionCalibrated(
                trial_id=pred.trial_id,
                probs_calibrated=calibrated_probs.tolist(),
                pred_class=calibrated_class,
                confidence_calibrated=float(max_calibrated_prob),
                abstained=abstained,
                version="1.0"
            )
            
            calibrated_predictions.append(calibrated_pred)
        
        return calibrated_predictions
    
    def fit_transform(self, 
                     predictions: List[Prediction], 
                     true_labels: List[str]) -> List[PredictionCalibrated]:
        """
        Fit calibration and transform predictions in one step.
        
        Parameters
        ----------
        predictions : List[Prediction]
            Predictions to calibrate
        true_labels : List[str]
            True labels for fitting
            
        Returns
        -------
        List[PredictionCalibrated]
            Calibrated predictions
        """
        self.fit(predictions, true_labels)
        return self.transform(predictions)
    
    def evaluate_calibration(self, 
                           predictions: List[Prediction], 
                           true_labels: List[str],
                           n_bins: int = 10) -> CalibrationMetrics:
        """
        Evaluate calibration quality before and after calibration.
        
        Parameters
        ----------
        predictions : List[Prediction]
            Original predictions
        true_labels : List[str]
            True labels
        n_bins : int
            Number of bins for calibration curve
            
        Returns
        -------
        CalibrationMetrics
            Calibration quality metrics
        """
        # Get calibrated predictions
        calibrated_preds = self.transform(predictions)
        
        # Calculate metrics for calibrated predictions
        cal_probs = np.array([pred.probs_calibrated for pred in calibrated_preds])
        
        # Create label mapping
        unique_labels = sorted(set(true_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        y_true = np.array([label_to_id[label] for label in true_labels])
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(cal_probs, y_true, n_bins)
        
        # Average Calibration Error (ACE)
        ace = self._calculate_ace(cal_probs, y_true, n_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(cal_probs, y_true, n_bins)
        
        # Brier Score
        brier_score = self._calculate_brier_score(cal_probs, y_true)
        
        # Negative Log-Likelihood
        nll = self._calculate_nll(cal_probs, y_true)
        
        return CalibrationMetrics(
            ece=float(ece),
            ace=float(ace),
            mce=float(mce),
            brier_score=float(brier_score),
            nll=float(nll),
            version="1.0"
        )
    
    def _calculate_ece(self, probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Get predictions in this confidence bin
            in_bin = (probs.max(axis=1) > bin_lower) & (probs.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (probs.argmax(axis=1)[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = probs.max(axis=1)[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_ace(self, probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> float:
        """Calculate Average Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ace = 0
        n_non_empty_bins = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs.max(axis=1) > bin_lower) & (probs.max(axis=1) <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = (probs.argmax(axis=1)[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = probs.max(axis=1)[in_bin].mean()
                ace += np.abs(avg_confidence_in_bin - accuracy_in_bin)
                n_non_empty_bins += 1
        
        return ace / n_non_empty_bins if n_non_empty_bins > 0 else 0
    
    def _calculate_mce(self, probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs.max(axis=1) > bin_lower) & (probs.max(axis=1) <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = (probs.argmax(axis=1)[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = probs.max(axis=1)[in_bin].mean()
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = max(mce, bin_error)
        
        return mce
    
    def _calculate_brier_score(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Brier Score."""
        # Convert to one-hot encoding
        n_classes = probs.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        
        # Brier score
        return np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))
    
    def _calculate_nll(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Negative Log-Likelihood."""
        log_probs = np.log(probs + 1e-15)
        return -np.mean(log_probs[np.arange(len(y_true)), y_true])
