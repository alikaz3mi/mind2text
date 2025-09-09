"""
Evaluation script for trained Mind2Text models.

This script evaluates trained models on test sets and generates
comprehensive reports including calibration metrics and visualizations.
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Mind2Text components
from mind2text.postprocessing import ProbabilityCalibrator, Evaluator, ReportGenerator
from mind2text.entities.modeling import Prediction, PredictionCalibrated
from mind2text.entities.reports import Report, ClassificationMetrics, CalibrationMetrics

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Mind2Text models")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--test_data_path", type=str, required=True,
                       help="Path to test data")
    
    # Evaluation arguments
    parser.add_argument("--calibration_method", type=str, default="temperature",
                       choices=["temperature", "isotonic", "none"],
                       help="Calibration method to use")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--n_bins", type=int, default=15,
                       help="Number of bins for calibration plots")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                       help="Output directory for results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save individual predictions")
    parser.add_argument("--generate_plots", action="store_true",
                       help="Generate calibration and performance plots")
    
    return parser.parse_args()

def load_model_and_artifacts(model_path: str):
    """Load trained model and preprocessing artifacts."""
    model_path = Path(model_path)
    
    logger.info(f"🤖 Loading model from {model_path}")
    
    # Load model configuration
    with open(model_path / "args.json", 'r') as f:
        model_config = json.load(f)
    
    # Load preprocessing artifacts
    from mind2text.algorithm import EEGTokenizer, FeatureBinner
    
    tokenizer = EEGTokenizer()
    tokenizer.load_vocabulary(str(model_path / "vocabulary.json"))
    
    binner = FeatureBinner()
    binner.load_binning_rules(str(model_path / "binning_rules.json"))
    
    # Load model based on type
    model_type = model_config['model_type']
    
    if model_type == 'cnn':
        from mind2text.models import CNNBaseline
        model = CNNBaseline.load(str(model_path / "model"))
        
    elif model_type == 'svm':
        from mind2text.models import SVMBaseline
        model = SVMBaseline.load(str(model_path / "model"))
        
    elif model_type in ['distilgpt2', 'gpt2']:
        from mind2text.models import LLMClassifier
        model = LLMClassifier.load(str(model_path / "model"))
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer, binner, model_config

def load_test_data(test_data_path: str):
    """Load test data."""
    test_data_path = Path(test_data_path)
    
    if test_data_path.suffix == '.json':
        # Load preprocessed test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        token_sequences = test_data['token_sequences']
        labels = test_data['labels']
        
    else:
        # Load raw EEG data and preprocess
        # This would use the same preprocessing pipeline as training
        raise NotImplementedError("Raw EEG evaluation not implemented yet")
    
    return token_sequences, labels

def generate_predictions(model, tokenizer, token_sequences, labels, model_config):
    """Generate predictions from model."""
    logger.info("🔮 Generating predictions...")
    
    # Encode sequences
    encoded_data = tokenizer.encode_batch(
        token_sequences, 
        max_length=model_config.get('max_length', 512)
    )
    
    # Create label mapping
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    predictions = []
    
    # Get model predictions
    if hasattr(model, 'predict_proba'):
        # For models that return probabilities
        probs = model.predict_proba(np.array(encoded_data['input_ids']))
        
        for i, (seq, true_label, prob) in enumerate(zip(token_sequences, labels, probs)):
            pred_class_id = np.argmax(prob)
            pred_class = id_to_label[pred_class_id]
            
            # Create logits from probabilities (approximate)
            logits = np.log(prob + 1e-8)
            
            prediction = Prediction(
                trial_id=f"test_{i:04d}",
                logits=logits.tolist(),
                probs=prob.tolist(),
                pred_class=pred_class,
                true_class=true_label,
                rationale_text=None,
                version="1.0"
            )
            predictions.append(prediction)
            
    else:
        # For models that only return class predictions
        pred_classes = model.predict(np.array(encoded_data['input_ids']))
        
        for i, (seq, true_label, pred_class_id) in enumerate(zip(token_sequences, labels, pred_classes)):
            pred_class = id_to_label[pred_class_id]
            
            # Create uniform probabilities (no confidence info available)
            n_classes = len(unique_labels)
            probs = np.ones(n_classes) / n_classes
            probs[pred_class_id] = 0.7  # Give some confidence to predicted class
            probs = probs / np.sum(probs)  # Renormalize
            
            logits = np.log(probs + 1e-8)
            
            prediction = Prediction(
                trial_id=f"test_{i:04d}",
                logits=logits.tolist(),
                probs=probs.tolist(),
                pred_class=pred_class,
                true_class=true_label,
                rationale_text=None,
                version="1.0"
            )
            predictions.append(prediction)
    
    logger.info(f"✅ Generated {len(predictions)} predictions")
    return predictions, unique_labels

def apply_calibration(predictions: List[Prediction], method: str, 
                     val_predictions: Optional[List[Prediction]] = None):
    """Apply probability calibration to predictions."""
    if method == "none":
        # Return predictions as calibrated (no calibration)
        calibrated_predictions = []
        for pred in predictions:
            cal_pred = PredictionCalibrated(
                trial_id=pred.trial_id,
                probs_calibrated=pred.probs,
                pred_class=pred.pred_class,
                true_class=pred.true_class,
                abstained=False,
                version="1.0"
            )
            calibrated_predictions.append(cal_pred)
        return calibrated_predictions, None
    
    logger.info(f"🎯 Applying {method} calibration...")
    
    calibrator = ProbabilityCalibrator(method=method)
    
    if val_predictions is not None:
        # Use validation set to fit calibration
        calibrator.fit(val_predictions)
    else:
        # Use test set to fit calibration (not recommended but sometimes necessary)
        logger.warning("⚠️  Using test set for calibration - results may be optimistic")
        calibrator.fit(predictions)
    
    # Apply calibration
    calibrated_predictions = []
    for pred in predictions:
        cal_pred = calibrator.calibrate_prediction(pred)
        calibrated_predictions.append(cal_pred)
    
    calibration_params = calibrator.get_calibration_params()
    return calibrated_predictions, calibration_params

def evaluate_predictions(predictions: List[PredictionCalibrated], 
                        class_names: List[str]) -> Tuple[ClassificationMetrics, CalibrationMetrics]:
    """Evaluate calibrated predictions."""
    logger.info("📊 Evaluating predictions...")
    
    evaluator = Evaluator()
    
    # Calculate classification metrics
    classification_metrics = evaluator.calculate_classification_metrics(predictions)
    
    # Calculate calibration metrics
    calibration_metrics = evaluator.calculate_calibration_metrics(predictions)
    
    return classification_metrics, calibration_metrics

def generate_plots(predictions: List[PredictionCalibrated], 
                  classification_metrics: ClassificationMetrics,
                  calibration_metrics: CalibrationMetrics,
                  output_dir: Path, n_bins: int = 15):
    """Generate evaluation plots."""
    logger.info("📈 Generating plots...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(classification_metrics.confusion.matrix)
    labels = classification_metrics.confusion.labels
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Calibration Plot (Reliability Diagram)
    plt.figure(figsize=(8, 6))
    
    # Extract probabilities and correctness
    probs = []
    correct = []
    
    for pred in predictions:
        max_prob = max(pred.probs_calibrated)
        is_correct = pred.pred_class == pred.true_class
        probs.append(max_prob)
        correct.append(is_correct)
    
    probs = np.array(probs)
    correct = np.array(correct)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            bin_sizes.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
            bin_sizes.append(0)
    
    # Plot calibration curve
    plt.bar(confidences, accuracies, width=1.0/n_bins, alpha=0.7, 
            edgecolor='black', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add ECE text
    plt.text(0.6, 0.2, f'ECE: {calibration_metrics.ece:.4f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "calibration_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class F1 scores
    plt.figure(figsize=(10, 6))
    classes = list(classification_metrics.per_class_f1.keys())
    f1_scores = list(classification_metrics.per_class_f1.values())
    
    bars = plt.bar(classes, f1_scores, alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "per_class_f1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confidence histogram
    plt.figure(figsize=(8, 6))
    plt.hist(probs, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Max Predicted Probability')
    plt.ylabel('Number of Predictions')
    plt.title('Confidence Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Saved plots to {plots_dir}")

def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / f"eval_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔬 Starting evaluation - output: {run_output_dir}")
    
    try:
        # Load model and artifacts
        model, tokenizer, binner, model_config = load_model_and_artifacts(args.model_path)
        
        # Load test data
        token_sequences, labels = load_test_data(args.test_data_path)
        
        # Generate predictions
        predictions, class_names = generate_predictions(
            model, tokenizer, token_sequences, labels, model_config
        )
        
        # Apply calibration
        calibrated_predictions, calibration_params = apply_calibration(
            predictions, args.calibration_method
        )
        
        # Evaluate predictions
        classification_metrics, calibration_metrics = evaluate_predictions(
            calibrated_predictions, class_names
        )
        
        # Create report
        report_generator = ReportGenerator()
        report = report_generator.create_evaluation_report(
            run_id=f"eval_{timestamp}",
            classification_metrics=classification_metrics,
            calibration_metrics=calibration_metrics,
            model_type=model_config['model_type'],
            dataset_split="test"
        )
        
        # Save results
        with open(run_output_dir / "report.json", 'w') as f:
            json.dump(report.model_dump(), f, indent=2)
        
        if args.save_predictions:
            predictions_data = {
                'predictions': [pred.model_dump() for pred in calibrated_predictions]
            }
            with open(run_output_dir / "predictions.json", 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        if calibration_params:
            with open(run_output_dir / "calibration_params.json", 'w') as f:
                json.dump(calibration_params.model_dump(), f, indent=2)
        
        # Generate plots
        if args.generate_plots:
            generate_plots(
                calibrated_predictions, classification_metrics, 
                calibration_metrics, run_output_dir, args.n_bins
            )
        
        # Print summary
        logger.info("📋 Evaluation Summary:")
        logger.info(f"   • Model: {model_config['model_type']}")
        logger.info(f"   • Test samples: {len(calibrated_predictions)}")
        logger.info(f"   • Accuracy: {classification_metrics.accuracy:.4f}")
        logger.info(f"   • Macro F1: {classification_metrics.macro_f1:.4f}")
        logger.info(f"   • ECE: {calibration_metrics.ece:.4f}")
        logger.info(f"   • Brier Score: {calibration_metrics.brier_score:.4f}")
        logger.info(f"   • Output: {run_output_dir}")
        
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
