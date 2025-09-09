"""
Training script for Mind2Text LLM and baseline models.

This script demonstrates how to:
1. Load and preprocess EEG data
2. Create symbolic token sequences  
3. Train LLM classifiers with LoRA
4. Train baseline models for comparison
5. Evaluate and compare performance
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Mind2Text components
from mind2text.preprocessing import EEGDataLoader, FeatureExtractor, SignalProcessor
from mind2text.algorithm import FeatureBinner, SymbolicEncoder, EEGTokenizer
from mind2text.entities.modeling import ModelConfig, TrainingRun
from mind2text.entities.reports import Report, ClassificationMetrics, CalibrationMetrics, ConfusionMatrix

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mind2Text models")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/ds004148",
                       help="Path to dataset")
    parser.add_argument("--n_subjects", type=int, default=10,
                       help="Number of subjects to use")
    parser.add_argument("--sessions", nargs="+", default=["session1"],
                       help="Sessions to use")
    parser.add_argument("--tasks", nargs="+", 
                       default=["memory", "mathematic", "music", "eyesopen", "eyesclosed"],
                       help="Tasks to include")
    
    # Feature extraction arguments
    parser.add_argument("--segment_length", type=float, default=4.0,
                       help="Segment length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap between segments")
    parser.add_argument("--n_bins", type=int, default=3,
                       help="Number of bins for feature discretization")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="distilgpt2",
                       choices=["distilgpt2", "gpt2", "cnn", "svm"],
                       help="Model type to train")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for LLM training")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/training",
                       help="Output directory")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name (auto-generated if None)")
    parser.add_argument("--save_model", action="store_true",
                       help="Save trained model")
    
    return parser.parse_args()

def load_and_preprocess_data(args) -> Tuple[List, List]:
    """Load and preprocess EEG data."""
    logger.info("🧠 Loading EEG data...")
    
    # Initialize data loader
    loader = EEGDataLoader(
        data_path=args.data_path,
        subject_ids=[f"sub-{i:02d}" for i in range(1, args.n_subjects + 1)],
        verbose=False
    )
    
    # Load trials
    all_trials = []
    subjects = loader.load_subjects_metadata()[:args.n_subjects]
    
    for subject in subjects:
        for session in args.sessions:
            for task in args.tasks:
                try:
                    raw, events, trial = loader.load_subject_session_data(
                        subject.subject_id, session, task
                    )
                    all_trials.append((raw, events, trial))
                except Exception as e:
                    logger.warning(f"Failed to load {subject.subject_id}_{session}_{task}: {e}")
    
    logger.info(f"📊 Loaded {len(all_trials)} trials from {len(subjects)} subjects")
    
    # Preprocess and extract features
    logger.info("🔧 Preprocessing and extracting features...")
    processor = SignalProcessor(sfreq=500.0)
    extractor = FeatureExtractor(sfreq=500.0)
    
    feature_vectors = []
    labels = []
    
    for raw, events, trial in all_trials:
        try:
            # Basic preprocessing
            raw_processed = processor.apply_basic_filters(raw)
            
            # Extract features
            fv_list, sf_list = extractor.extract_features_from_raw(
                raw_processed,
                trial.trial_id,
                segment_length=args.segment_length,
                overlap=args.overlap
            )
            
            feature_vectors.extend(fv_list)
            task_labels = [trial.task] * len(fv_list)
            labels.extend(task_labels)
            
        except Exception as e:
            logger.error(f"Failed to process {trial.trial_id}: {e}")
    
    logger.info(f"📈 Extracted {len(feature_vectors)} feature vectors")
    
    # Print label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    logger.info(f"📋 Label distribution: {label_dist}")
    
    return feature_vectors, labels

def create_token_sequences(feature_vectors, labels, args):
    """Create symbolic token sequences."""
    logger.info("🔤 Creating symbolic tokens...")
    
    # Discretize features
    binner = FeatureBinner(n_bins=args.n_bins, strategy='quantile')
    binned_features_list = binner.fit_transform(feature_vectors)
    
    # Encode to tokens
    encoder = SymbolicEncoder(
        include_channel_info=True,
        include_band_info=True
    )
    
    token_sequences = []
    for i, (fv, binned_features) in enumerate(zip(feature_vectors, binned_features_list)):
        tokens = encoder.encode_binned_features(
            binned_features,
            fv.trial_id,
            vocab_version=f"training_{args.run_name}"
        )
        # Add special tokens
        tokens_with_special = encoder.add_special_tokens(tokens)
        token_sequences.append(tokens_with_special)
    
    # Build vocabulary
    tokenizer = EEGTokenizer(max_vocab_size=10000)
    tokenizer.build_vocabulary(token_sequences, min_frequency=2)
    
    vocab_summary = tokenizer.get_vocabulary_summary()
    logger.info(f"📖 Built vocabulary with {vocab_summary['vocab_size']} tokens")
    
    return token_sequences, tokenizer, binner, encoder

def split_data(token_sequences, labels, test_size=0.2, val_size=0.1, seed=42):
    """Split data into train/val/test sets."""
    np.random.seed(seed)
    
    n_samples = len(token_sequences)
    indices = np.random.permutation(n_samples)
    
    test_idx = int(n_samples * test_size)
    val_idx = int(n_samples * (test_size + val_size))
    
    test_indices = indices[:test_idx]
    val_indices = indices[test_idx:val_idx]
    train_indices = indices[val_idx:]
    
    train_sequences = [token_sequences[i] for i in train_indices]
    val_sequences = [token_sequences[i] for i in val_indices]
    test_sequences = [token_sequences[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    logger.info(f"📊 Data split - Train: {len(train_sequences)}, "
               f"Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    return (train_sequences, val_sequences, test_sequences,
            train_labels, val_labels, test_labels)

def train_baseline_model(train_data, train_labels, val_data, val_labels, 
                        test_data, test_labels, tokenizer, args):
    """Train baseline CNN or SVM model."""
    logger.info(f"🤖 Training {args.model_type} baseline...")
    
    # Encode sequences to numerical format
    train_encoded = tokenizer.encode_batch(train_data, max_length=args.max_length)
    val_encoded = tokenizer.encode_batch(val_data, max_length=args.max_length)
    test_encoded = tokenizer.encode_batch(test_data, max_length=args.max_length)
    
    X_train = np.array(train_encoded['input_ids'])
    X_val = np.array(val_encoded['input_ids'])
    X_test = np.array(test_encoded['input_ids'])
    
    # Create label mapping
    unique_labels = sorted(set(train_labels + val_labels + test_labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    
    y_train = np.array([label_to_id[label] for label in train_labels])
    y_val = np.array([label_to_id[label] for label in val_labels])
    y_test = np.array([label_to_id[label] for label in test_labels])
    
    if args.model_type == 'cnn':
        from mind2text.models import CNNBaseline
        
        model = CNNBaseline(
            input_dim=X_train.shape[1],
            n_classes=len(unique_labels),
            hidden_dim=128
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True
        )
        
        # Evaluate
        test_accuracy = model.evaluate(X_test, y_test)
        test_predictions = model.predict(X_test)
        
    elif args.model_type == 'svm':
        from mind2text.models import SVMBaseline
        
        model = SVMBaseline(kernel='rbf', C=1.0)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        test_accuracy = model.score(X_test, y_test)
        test_predictions = model.predict(X_test)
    
    logger.info(f"🎯 {args.model_type.upper()} Test Accuracy: {test_accuracy:.4f}")
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, test_predictions)
    
    confusion_matrix_entity = ConfusionMatrix(
        labels=unique_labels,
        matrix=cm.tolist(),
        version="1.0"
    )
    
    # Calculate detailed metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(y_test, test_predictions)
    macro_f1 = f1_score(y_test, test_predictions, average='macro')
    weighted_f1 = f1_score(y_test, test_predictions, average='weighted')
    
    per_class_precision = precision_score(y_test, test_predictions, average=None)
    per_class_recall = recall_score(y_test, test_predictions, average=None)
    per_class_f1 = f1_score(y_test, test_predictions, average=None)
    
    classification_metrics = ClassificationMetrics(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class_precision={unique_labels[i]: float(per_class_precision[i]) for i in range(len(unique_labels))},
        per_class_recall={unique_labels[i]: float(per_class_recall[i]) for i in range(len(unique_labels))},
        per_class_f1={unique_labels[i]: float(per_class_f1[i]) for i in range(len(unique_labels))},
        version="1.0"
    )
    
    return model, classification_metrics, confusion_matrix_entity

def create_training_report(args, classification_metrics, confusion_matrix_entity, 
                         run_start_time):
    """Create comprehensive training report."""
    
    # Create dummy calibration metrics for baseline models
    calibration_metrics = CalibrationMetrics(
        ece=0.0,  # Not applicable for baseline models
        ace=0.0,
        mce=0.0,
        brier_score=0.0,
        nll=0.0,
        version="1.0"
    )
    
    report = Report(
        run_id=args.run_name,
        classification_metrics=classification_metrics,
        calibration_metrics=calibration_metrics,
        confusion=confusion_matrix_entity,
        dataset_split="test",
        model_type=args.model_type,
        evaluation_timestamp=datetime.now().isoformat(),
        version="1.0"
    )
    
    return report

def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.model_type}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"🚀 Starting training run: {args.run_name}")
    run_start_time = datetime.now()
    
    try:
        # Load and preprocess data
        feature_vectors, labels = load_and_preprocess_data(args)
        
        # Create token sequences
        token_sequences, tokenizer, binner, encoder = create_token_sequences(
            feature_vectors, labels, args
        )
        
        # Split data
        (train_sequences, val_sequences, test_sequences,
         train_labels, val_labels, test_labels) = split_data(
            token_sequences, labels, seed=args.seed
        )
        
        # Save preprocessing artifacts
        tokenizer.save_vocabulary(str(output_dir / "vocabulary.json"))
        binner.save_binning_rules(str(output_dir / "binning_rules.json"))
        
        # Train model based on type
        if args.model_type in ['cnn', 'svm']:
            model, classification_metrics, confusion_matrix_entity = train_baseline_model(
                train_sequences, train_labels, val_sequences, val_labels,
                test_sequences, test_labels, tokenizer, args
            )
            
        else:
            # TODO: Implement LLM training
            logger.info("LLM training not yet implemented in this script")
            return
        
        # Create and save report
        report = create_training_report(
            args, classification_metrics, confusion_matrix_entity, run_start_time
        )
        
        # Save report
        with open(output_dir / "report.json", 'w') as f:
            json.dump(report.model_dump(), f, indent=2)
        
        # Save model if requested
        if args.save_model and hasattr(model, 'save'):
            model.save(str(output_dir / "model"))
        
        # Print summary
        logger.info("📊 Training Summary:")
        logger.info(f"   • Model: {args.model_type}")
        logger.info(f"   • Accuracy: {classification_metrics.accuracy:.4f}")
        logger.info(f"   • Macro F1: {classification_metrics.macro_f1:.4f}")
        logger.info(f"   • Output: {output_dir}")
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
