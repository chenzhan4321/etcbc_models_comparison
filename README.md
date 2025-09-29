# SyriacML: Neural Networks for Morphological Analysis

![Version](https://img.shields.io/badge/version-12.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Open%20Source-green.svg)

## 📋 Project Overview

Deep learning project for Syriac morphological analysis supporting 13 neural network architectures. Core task: sequence labeling — given Syriac character sequences, predict morphological tags for each position.

### Key Features
- **3 Core Architectures**: Transformer, MDLM, and LSTM for different use cases
- **5 Other Model Types**: BERT, Mamba, RetNet, Switch, and RWKV series
- **Dual Dataset Support**: S2 (default) and S4 (extended) datasets
- **Complete Pipeline**: Training, validation, and data processing system

## ⚠️ Dataset System

Project supports two dataset configurations with **clear separation**:

| Configuration | Training Dataset | Test Dataset | Training Size | Path |
|---------------|------------------|--------------|---------------|------|
| **S2-on-S2** (default) | S2 | S2 | 50,649 lines | `data/s2_on_s2/` |
| **S4-on-S2** (extended training) | S4 | S2 | 97,048 lines | `data/s4_on_s2/` |

### Dataset Details
- **S2-on-S2**: Standard configuration, same dataset for training and testing
- **S4-on-S2**: Extended training data (S4) with consistent test data (S2) for fair comparison
- **Validation & Test**: Both configurations use identical validation (10,964) and test (10,869) sets from S2
- **Classes**: S2 has 309 classes, S4 training has ~350 classes

## 🏆 Current SOTA Results (Character Error Rate - CER)

| Model | S2-on-S2 | S4-on-S2 | Best Configuration |
|-------|----------|----------|-------------------|
| **MDLM** | **1.61%** | **1.81%** | **🥇 SOTA - Best overall accuracy** |
| **Transformer** | 1.95% | 1.99% | 🥈 Solid baseline performance |
| **LSTM** | 1.56% | 1.67% | 🥉 Most efficient training |
| **Encoder** | - | 98.44% | Experimental configuration |

### Key Performance Insights
- **🏆 MDLM leads with 1.61% CER** on S2-on-S2 configuration
- **📈 Extended training (S4-on-S2)** shows marginal improvements across models
- **⚡ LSTM** achieves competitive results with fastest training speed
- **🎯 All core models** achieve sub-2% character error rates

## 🏗️ Project Architecture

### Directory Structure
- **`models/`**: Model definition package (pure code, no training results)
- **`outputs/`**: Training results and model files storage directory
- **`data/`**: Training data and data processing scripts
- **`data_processing_tools/`**: Data analysis and processing utilities
- **`runs/`**: TensorBoard logs (excluded from git)
- **`logs/`**: Training logs (excluded from git)

## 📊 Core Model Architectures

The project focuses on three primary architectures, each representing different paradigms:

### 🏗️ Transformer (Encoder-Only)
**Standard attention-based sequence labeling model**
- **Architecture**: Multi-head self-attention with positional encoding
- **Complexity**: O(n²) time, O(n²) memory
- **Vocabulary**: 26 characters (basic Syriac character set)
- **Key Features**:
  - Parallel processing of sequences
  - Global attention mechanism
  - Strong baseline performance
- **Use Cases**: Standard benchmark, production baseline
- **Training**: `python train.py --model_type transformer --epochs 50`

### 🎭 MDLM (Masked Diffusion Language Model)
**Discrete diffusion model for sequence labeling**
- **Architecture**: Transformer-based diffusion with discrete masking
- **Complexity**: O(n²) time with diffusion steps
- **Vocabulary**: 40 characters (extended character set with digits and special tokens)
- **Key Features**:
  - Iterative refinement through diffusion
  - Handles uncertainty in morphological analysis
  - Generation capabilities
- **Use Cases**: High-quality morphological analysis, uncertainty quantification
- **Training**: `python train.py --model_type mdlm --epochs 80`

### 🔄 LSTM (Long Short-Term Memory)
**Recurrent architecture for sequential processing**
- **Architecture**: Bidirectional LSTM with attention
- **Complexity**: O(n) time, O(n) memory
- **Vocabulary**: 26 characters (basic Syriac character set)
- **Key Features**:
  - Sequential processing
  - Memory efficiency
  - Fast inference
- **Use Cases**: Resource-constrained environments, fast inference
- **Training**: `python train.py --model_type lstm --epochs 100`

### Core Models Comparison

| Model | Complexity | Memory | Vocab Size | Best CER | Best For |
|-------|------------|--------|------------|----------|-----------|
| **Transformer** | O(n²) | High | 26 | 1.95% | Standard baseline, benchmarking |
| **MDLM** | O(n²) + diffusion | High | 40 | **1.61%** | **🏆 SOTA accuracy, uncertainty quantification** |
| **LSTM** | O(n) | Low | 26 | **1.56%** | **⚡ Fast training, production deployment** |

## 🔧 Other Models

The project also supports additional architectures for experimentation:

| Architecture | Key Innovation | Use Case |
|--------------|----------------|----------|
| **BERT** | Bidirectional encoder | Contextual understanding |
| **Mamba** | State-space models (various sizes) | Efficient long-range modeling |
| **RetNet** | Retention mechanism | Real-time applications |
| **Switch** | Mixture of experts | Large-scale multi-task |
| **RWKV** | Linear attention (various sizes) | Long sequences, production |

**Quick Start with Other Models:**
```bash
# Try other architectures
python train.py --model_type bert --epochs 30
python train.py --model_type mamba --epochs 50
python train.py --model_type rwkv7 --epochs 50
python train.py --model_type retnet --epochs 60
python train.py --model_type switch --epochs 40
```

## 🚀 Core Commands

### Basic Training

#### Core Models (Recommended Starting Point)
```bash
# S2-on-S2 Configuration (Default)
# Transformer - Standard attention-based baseline
python train.py --model_type transformer --epochs 50 --batch_size 32 --data_dir data/s2_on_s2

# MDLM - High-quality diffusion model
python train.py --model_type mdlm --epochs 80 --batch_size 16 --data_dir data/s2_on_s2

# LSTM - Fast and efficient recurrent model
python train.py --model_type lstm --epochs 100 --batch_size 64 --data_dir data/s2_on_s2

# S4-on-S2 Configuration (More Training Data)
# Train on larger S4 dataset, test on consistent S2 dataset
python train.py --model_type transformer --epochs 50 --batch_size 32 --data_dir data/s4_on_s2
python train.py --model_type mdlm --epochs 80 --batch_size 16 --data_dir data/s4_on_s2
```

#### Other Architectures (Experimental)
```bash
# Try other models
python train.py --model_type bert --epochs 30 --batch_size 32
python train.py --model_type mamba --epochs 50 --batch_size 32
python train.py --model_type rwkv7 --batch_size 16 --epochs 100
python train.py --model_type retnet --epochs 60 --batch_size 32
```

#### Hyperparameter Optimization
```bash
# Focus on core models first
python train_hpo.py --models transformer,mdlm,lstm --n_trials 50

# Then explore other architectures
python train_hpo.py --models rwkv7,mamba,retnet --n_trials 30
```

### Model Testing & Inference

#### Core Models Evaluation
```bash
# Test core models accuracy
python data_processing_tools/accuracy_analysis_system/accuracy_analyzer.py
python data_processing_tools/levenshtein/compare_final_units.py

# Compare core models performance
python data_processing_tools/compare_word_level_correctness/verify_accuracy_simple.py

# Quick core models comparison
python train_hpo.py --compare_models --compare_groups light medium --compare_mode quick
```

### Environment Verification
```bash
# System environment check
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from models import list_available_models; print('Supported models:', list_available_models())"

# Model registry check
python -c "from models import MODEL_REGISTRY; print('Registered models:', list(MODEL_REGISTRY.keys()))"
```

## ⚙️ Model Usage

### Core Models Factory Usage
```python
from models.model_factory import create_model
from models import get_model_class

# Core models - recommended for most users
# Transformer (encoder-only)
model = create_model('transformer', vocab_size=26, num_classes=309)

# MDLM (diffusion model)
model = create_model('mdlm', vocab_size=40, num_classes=309)

# LSTM (recurrent model)
model = create_model('lstm', vocab_size=26, num_classes=309)

# Other architectures
model = create_model('rwkv7', vocab_size=26, num_classes=309)
model = create_model('mamba', vocab_size=26, num_classes=309)

# Method 2: Direct model class access
TransformerClass = get_model_class('transformer')
model = TransformerClass(vocab_size=26, num_classes=309)
```

## 📊 Data Processing System

### Character Encoding System
- **Basic Character Set**: 26 Syriac characters (1 space + 25 letters)
- **Extended Character Set**: 40 characters (basic 26 + digits 0-9 + 4 special tokens)
- **Model Compatibility**:
  - **Core models**: Transformer and LSTM use 26 basic characters
  - **MDLM model**: Uses 40 extended characters (includes digits and special tokens)
  - **Other models**: Most use 26 characters, check model documentation

### Data Post-processing Pipeline
```
Model Output (.out.original)
    ↓
data_processing_tools/decompose/01_out_original_reducer.py
    ↓
Reduced Format (.out.reduced)
    ↓
data_processing_tools/decompose/03_convert_reduced_to_final.py
    ↓
Final Format (.out.final)
    ↓
data_processing_tools/decompose/04_convert_final_to_out.py
    ↓
Standard Output (.out)
```

### Data Directory Structure
- **Core Data Files**: `train.in`, `val.in`, `test.in` (inputs) and corresponding `.out` files (labels)
- **Data Processing Scripts**:
  - `decompose/`: Data post-processing pipeline scripts
  - `accuracy_analysis_system/`: Accuracy analysis tools
  - `levenshtein/`: String distance analysis for morphological similarity
- **Extended Dataset**: `data_comparison/` directory contains S4 dataset

### Accuracy Analysis Tools
```bash
# Overall accuracy analysis
python data_processing_tools/accuracy_analysis_system/accuracy_analyzer.py

# String distance-based analysis
python data_processing_tools/levenshtein/compare_final_units.py
```

## 🎯 Trained Model Examples

Based on models already trained in the `outputs/` directory:

### S2-on-S2 Configuration (Standard)
```bash
# Transformer S2-on-S2 training (✅ Completed)
python train.py --model_type transformer --data_dir data/s2_on_s2 \
    --epochs 30 --batch_size 128 --learning_rate 0.0001 \
    --embedding_size 512 --num_heads 8 --dropout 0.1

# LSTM S2-on-S2 training (✅ Completed)
python train.py --model_type lstm --data_dir data/s2_on_s2 \
    --epochs 50 --batch_size 64

# MDLM S2-on-S2 training (✅ Completed)
python train.py --model_type mdlm --data_dir data/s2_on_s2 \
    --epochs 80 --batch_size 16
```

### S4-on-S2 Configuration (Extended Training)
```bash
# Transformer S4-on-S2 training (✅ Completed)
python train.py --model_type transformer --data_dir data/s4_on_s2 \
    --epochs 50 --batch_size 32

# LSTM S4-on-S2 training (✅ Completed)
python train.py --model_type lstm --data_dir data/s4_on_s2 \
    --epochs 50 --batch_size 64

# MDLM S4-on-S2 training (✅ Completed)
python train.py --model_type mdlm --data_dir data/s4_on_s2 \
    --epochs 80 --batch_size 16

# Encoder-only S4-on-S2 training (✅ Completed)
python train.py --model_type transformer --data_dir data/s4_on_s2 \
    --model_variant encoder_only --epochs 40
```

### Model Output Locations
- **transformer_s2_on_s2/**: Standard transformer results
- **lstm_s2_on_s2/**: LSTM baseline results
- **mdlm_s2_on_s2/**: MDLM diffusion model results
- **transformer_s4_on_s2/**: Extended training transformer results
- **lstm_s4_on_s2/**: Extended training LSTM results
- **mdlm_s4_on_s2/**: Extended training MDLM results

## 🛠 Performance Tuning

### Accuracy Targets (Character Error Rate)
- **Current SOTA**: 1.56-1.61% CER (LSTM and MDLM models)
- **Competitive Range**: 1.6-2.0% CER (all core models)
- **Production Ready**: Sub-2% CER consistently achieved

### Recommended Configurations
```bash
# Memory-constrained environment
python train.py --model_type lstm --batch_size 64 --epochs 100

# Balanced performance
python train.py --model_type transformer --batch_size 32 --epochs 50

# High-quality results
python train.py --model_type mdlm --batch_size 16 --epochs 80
```

### Performance Monitoring
```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# View training logs
tail -f logs/train_*.log
```

## 🔍 Troubleshooting

### Environment Issues
```bash
# Dependency installation
pip install -r requirements.txt

# Environment check
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Data integrity
ls data/train.in data/val.in data/test.in  # Confirm data files exist
```

### Model Issues
```bash
# Check model registration
python -c "from models import MODEL_REGISTRY; print(list(MODEL_REGISTRY.keys()))"

# Test model creation
python -c "from models.model_factory import create_model; m=create_model('transformer', 26, 309); print('✓ Transformer available')"

# vocab_size mismatch issues
python -c "from models.core import get_vocab_size; print(f'Current vocab_size: {get_vocab_size()}')"
```

### Common Error Solutions
1. **Out of Memory**: Reduce batch_size, use efficient model versions
2. **CUDA Errors**: Confirm GPU drivers, check torch.cuda.is_available()
3. **Model Loading Failure**: Check vocab_size consistency with training (26 vs 40)
4. **Accuracy Anomalies**: Verify dataset selection (S2/S4), check patterns.csv path

## 📚 Key Considerations

1. **Dataset Consistency**: Ensure training, validation, testing use the same dataset (S2 or S4)
2. **vocab_size Matching**: Traditional models use 26, MDLM uses 40
3. **patterns.csv Correspondence**: Data post-processing uses correct patterns.csv file
4. **Directory Structure**: Model files saved in outputs/, code in models/, logs in logs/, TensorBoard in runs/
5. **Memory Management**: Adjust batch_size and model version selection based on GPU memory

## 📄 License

This project follows open source license requirements.

---

**Version**: 12.0.0
**Last Updated**: September 2025
**Architecture Support**: 13 neural network architectures (3 core + 10 other models)
**Focus**: Syriac morphological analysis with sequence labeling