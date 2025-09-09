"""
Basic Usage Example for Mind2Text

This example demonstrates the complete pipeline from EEG data to 
cognitive state classification with explanations.
"""

import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Mind2Text components
from mind2text.preprocessing import EEGDataLoader, FeatureExtractor, SignalProcessor
from mind2text.algorithm import FeatureBinner, SymbolicEncoder, EEGTokenizer
from mind2text.models import LLMClassifier, CNNBaseline
from mind2text.entities.modeling import ModelConfig

def main():
    """Run the basic Mind2Text pipeline."""
    
    # Configuration
    data_path = "data/ds004148"
    subject_ids = ["sub-01", "sub-02", "sub-03"]  # Start with 3 subjects
    output_dir = Path("outputs/basic_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧠 Starting Mind2Text basic example")
    
    # Step 1: Load EEG Data
    logger.info("📁 Loading EEG data...")
    loader = EEGDataLoader(
        data_path=data_path,
        subject_ids=subject_ids,
        verbose=True
    )
    
    # Load a few trials for demonstration
    all_trials = []
    for subject_id in subject_ids[:2]:  # Use first 2 subjects
        for session in ["session1"]:  # Use first session
            for task in ["memory", "mathematic", "music"]:  # Use 3 tasks
                try:
                    raw, events, trial = loader.load_subject_session_data(
                        subject_id, session, task
                    )
                    all_trials.append((raw, events, trial))
                    logger.info(f"✅ Loaded {trial.trial_id}")
                except Exception as e:
                    logger.warning(f"❌ Failed to load {subject_id}_{session}_{task}: {e}")
    
    logger.info(f"📊 Loaded {len(all_trials)} trials")
    
    # Step 2: Preprocess and Extract Features
    logger.info("🔧 Preprocessing and extracting features...")
    processor = SignalProcessor(sfreq=500.0)
    extractor = FeatureExtractor(sfreq=500.0)
    
    feature_vectors = []
    labels = []
    
    for raw, events, trial in all_trials:
        try:
            # Basic preprocessing
            raw_processed = processor.apply_basic_filters(raw)
            
            # Extract features using sliding windows
            fv_list, sf_list = extractor.extract_features_from_raw(
                raw_processed, 
                trial.trial_id,
                segment_length=4.0,
                overlap=0.5
            )
            
            # Add features and labels
            feature_vectors.extend(fv_list)
            task_labels = [trial.task] * len(fv_list)
            labels.extend(task_labels)
            
            logger.info(f"✅ Extracted {len(fv_list)} feature vectors from {trial.trial_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {trial.trial_id}: {e}")
    
    logger.info(f"📈 Total feature vectors: {len(feature_vectors)}")
    logger.info(f"📋 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Step 3: Create Symbolic Tokens
    logger.info("🔤 Creating symbolic tokens...")
    
    # Discretize features
    binner = FeatureBinner(n_bins=3, strategy='quantile')
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
            vocab_version="basic_example_v1"
        )
        token_sequences.append(tokens)
    
    # Add special tokens
    token_sequences_with_special = []
    for seq in token_sequences:
        seq_with_special = encoder.add_special_tokens(seq)
        token_sequences_with_special.append(seq_with_special)
    
    logger.info(f"🎯 Created {len(token_sequences_with_special)} token sequences")
    
    # Print sample token sequence
    if token_sequences_with_special:
        sample_seq = token_sequences_with_special[0]
        sample_text = encoder.tokens_to_text(sample_seq)
        logger.info(f"📝 Sample tokens: {sample_text[:200]}...")
    
    # Step 4: Build Vocabulary and Tokenizer
    logger.info("📚 Building vocabulary...")
    tokenizer = EEGTokenizer(max_vocab_size=5000)
    tokenizer.build_vocabulary(token_sequences_with_special, min_frequency=2)
    
    vocab_summary = tokenizer.get_vocabulary_summary()
    logger.info(f"📖 Vocabulary size: {vocab_summary['vocab_size']}")
    
    # Save vocabulary
    vocab_file = output_dir / "vocabulary.json"
    tokenizer.save_vocabulary(str(vocab_file))
    logger.info(f"💾 Saved vocabulary to {vocab_file}")
    
    # Step 5: Encode Sequences
    logger.info("🔢 Encoding sequences to numerical IDs...")
    encoded_batch = tokenizer.encode_batch(
        token_sequences_with_special,
        max_length=256,
        padding=True,
        truncation=True
    )
    
    logger.info(f"🎛️ Encoded {len(encoded_batch['input_ids'])} sequences")
    logger.info(f"🔢 Sequence length: {len(encoded_batch['input_ids'][0])}")
    
    # Step 6: Train Baseline Model (CNN)
    logger.info("🤖 Training CNN baseline...")
    
    # Convert encoded sequences to features for CNN
    # For simplicity, we'll use the token IDs as features
    X_cnn = np.array(encoded_batch['input_ids'])
    y_cnn = np.array([label for label in labels])
    
    # Create label mapping
    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    y_cnn_numeric = np.array([label_to_id[label] for label in labels])
    
    cnn_baseline = CNNBaseline(
        input_dim=X_cnn.shape[1],
        n_classes=len(unique_labels),
        hidden_dim=128
    )
    
    # Simple train/test split
    split_idx = int(0.8 * len(X_cnn))
    X_train, X_test = X_cnn[:split_idx], X_cnn[split_idx:]
    y_train, y_test = y_cnn_numeric[:split_idx], y_cnn_numeric[split_idx:]
    
    # Train CNN
    cnn_history = cnn_baseline.fit(X_train, y_train, epochs=5, verbose=True)
    
    # Evaluate CNN
    cnn_accuracy = cnn_baseline.evaluate(X_test, y_test)
    logger.info(f"🎯 CNN Baseline Accuracy: {cnn_accuracy:.3f}")
    
    # Step 7: Prepare for LLM Training (Configuration Only)
    logger.info("🚀 Preparing LLM configuration...")
    
    llm_config = ModelConfig(
        base_model="distilgpt2",
        lora_r=8,
        lora_alpha=16,
        lr=2e-4,
        batch_size=8,
        epochs=3,
        max_sequence_length=256,
        seed=42
    )
    
    logger.info(f"⚙️ LLM Config: {llm_config.base_model} with LoRA r={llm_config.lora_r}")
    
    # Note: Actual LLM training would require more data and compute
    # This is just showing the configuration
    
    # Step 8: Summary
    logger.info("📊 Pipeline Summary:")
    logger.info(f"   • Loaded {len(all_trials)} EEG trials")
    logger.info(f"   • Extracted {len(feature_vectors)} feature vectors")
    logger.info(f"   • Created {len(token_sequences)} token sequences")
    logger.info(f"   • Vocabulary size: {vocab_summary['vocab_size']}")
    logger.info(f"   • CNN accuracy: {cnn_accuracy:.3f}")
    logger.info(f"   • Ready for LLM training with {llm_config.base_model}")
    
    # Step 9: Save Results
    results = {
        "n_trials": len(all_trials),
        "n_features": len(feature_vectors),
        "n_tokens": len(token_sequences),
        "vocab_size": vocab_summary['vocab_size'],
        "cnn_accuracy": float(cnn_accuracy),
        "label_distribution": dict(zip(*np.unique(labels, return_counts=True))),
        "config": llm_config.model_dump()
    }
    
    import json
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Saved results to {results_file}")
    logger.info("✅ Basic example completed successfully!")

if __name__ == "__main__":
    main()
