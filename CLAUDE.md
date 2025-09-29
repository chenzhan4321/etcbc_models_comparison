# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Syriac morphological analysis deep learning project supporting 13 neural network architectures. Core task: sequence labeling — given Syriac character sequences, predict morphological tags for each position.

### Key Features
- **13 Model Architectures**: From classic Transformer to latest RWKV-7
- **Dual Dataset Support**: S2 (default) and S4 (extended) datasets
- **Linear Complexity**: RWKV-7 and similar architectures support efficient long sequence processing
- **Production Ready**: Complete training, validation, and deployment pipeline

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

## Core Commands

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

#### Advanced Architectures (Optional)
```bash
# Cutting-edge linear complexity models
python train.py --model_type rwkv7 --batch_size 16 --epochs 100
python train.py --model_type mamba --epochs 80 --batch_size 32
python train.py --model_type retnet --epochs 60 --batch_size 32
```

#### Hyperparameter Optimization
```bash
# Focus on core models first
python train_hpo.py --models transformer,mdlm,lstm --n_trials 50

# Then explore advanced architectures
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

## Project Architecture

### Directory Structure
- **`models/`**: Model definition package (pure code, no training results)
- **`outputs/`**: Training results and model files storage directory
- **`data/`**: Training data and data processing scripts
- **`data_processing_tools/`**: Data analysis and processing utilities
- **`runs/`**: TensorBoard logs (excluded from git)
- **`logs/`**: Training logs (excluded from git)

### Core Modules
- **`train.py`**: Main training script supporting all 13 model architectures
- **`train_hpo.py`**: Hyperparameter optimization training script
- **`models/`**: Model implementation package, managed via MODEL_REGISTRY
- **`models/data_utils.py`**: Data loading and processing (default S2 dataset)
- **`models/model_factory.py`**: Model factory for dynamic model instance creation
- **`models/config_manager.py`**: Unified configuration management system
- **`models/core.py`**: Device management, logging, vocabulary core tools
- **`models/components/`**: Reusable neural network components

### Core Model Architectures

The project focuses on three primary architectures, each representing different paradigms:

#### 🏗️ Transformer (Encoder-Only)
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

#### 🎭 MDLM (Masked Diffusion Language Model)
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

#### 🔄 LSTM (Long Short-Term Memory)
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

#### 🚀 Additional Architectures
**Other supported models (brief overview):**
- **Cutting-edge**: RWKV-7 (linear complexity), Mamba (state-space), RetNet (retention)
- **Enhanced**: BiMamba, Switch Transformer, various RWKV-7 variants
- **Classic**: BERT (bidirectional encoder)

*For detailed information on advanced architectures, see the cutting-edge models section below.*

### Cutting-Edge Architectures (Advanced Users)

The project also supports modern architectures for advanced users:

| Architecture | Key Innovation | Complexity | Use Case |
|--------------|----------------|------------|----------|
| **RWKV-7** | Linear attention, dynamic state evolution | O(n) | Long sequences, production |
| **Mamba** | State-space model, selective scanning | O(n) | Efficient long-range modeling |
| **RetNet** | Retention mechanism, parallel + recurrent | O(n) | Real-time applications |
| **BiMamba** | Bidirectional state-space | O(n) | Enhanced context understanding |
| **Switch** | Mixture of experts | O(n) | Large-scale multi-task |

**Quick Start with Advanced Models:**
```bash
# Try RWKV-7 for linear complexity
python train.py --model_type rwkv7 --epochs 50

# Compare with Mamba
python train.py --model_type mamba --epochs 50

# HPO comparison of advanced models
python train_hpo.py --models rwkv7,mamba,retnet --n_trials 20
```

### Model Factory Usage
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

# Advanced architectures
model = create_model('rwkv7', vocab_size=26, num_classes=309)
model = create_model('mamba', vocab_size=26, num_classes=309)

# Method 2: Direct model class access
TransformerClass = get_model_class('transformer')
model = TransformerClass(vocab_size=26, num_classes=309)
```

### Core Models Comparison

| Model | Complexity | Memory | Vocab Size | Best For |
|-------|------------|--------|------------|-----------|
| **Transformer** | O(n²) | High | 26 | Standard baseline, benchmarking |
| **MDLM** | O(n²) + diffusion | High | 40 | High-quality analysis, uncertainty |
| **LSTM** | O(n) | Low | 26 | Fast inference, limited resources |

## Data Processing System

### Character Encoding System
- **Basic Character Set**: 26 Syriac characters (1 space + 25 letters)
- **Extended Character Set**: 40 characters (basic 26 + digits 0-9 + 4 special tokens)
- **Model Compatibility**:
  - **Core models**: Transformer and LSTM use 26 basic characters
  - **MDLM model**: Uses 40 extended characters (includes digits and special tokens)
  - **Advanced models**: Most use 26 characters, check model documentation

### Data Post-processing Pipeline
```
Model Output (.out.original)
    ↓
data/decompose/01_out_original_reducer.py
    ↓
Reduced Format (.out.reduced)
    ↓
data/decompose/03_convert_reduced_to_final.py
    ↓
Final Format (.out.final)
    ↓
data/decompose/04_convert_final_to_out.py
    ↓
Standard Output (.out)
```

### Data Directory Structure
- **Core Data Files**: `train.in`, `val.in`, `test.in` (inputs) and corresponding `.out` files (labels)
- **Data Processing Scripts**:
  - `decompose/`: Data post-processing pipeline scripts
  - `accuracy_analysis_system/`: Accuracy analysis tools
  - `compare_point_correctness/`: Diacritical mark accuracy analysis
  - `compare_word_level_correctness/`: Word-level accuracy comparison
- **Extended Dataset**: `data_comparison/` directory contains S4 dataset

### Accuracy Analysis Tools
```bash
# Overall accuracy analysis
python data/accuracy_analysis_system/accuracy_analyzer.py

# Point-wise accuracy (diacritical marks)
python data/compare_point_correctness/compare_final_units.py

# Word-level accuracy comparison
python data/compare_word_level_correctness/verify_accuracy_simple.py
```

## Performance Tuning

### Accuracy Targets
- **Baseline**: 84% (current benchmark)
- **Target**: 95%+ (cutting-edge architectures)
- **Best**: 98%+ (ensemble models)

### Recommended Configurations
```bash
# High-performance RWKV-7 training
python train.py --model_type rwkv7_large \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --epochs 200 \
    --gradient_clip 1.0

# Quick validation configuration
python train.py --model_type rwkv7_efficient \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --epochs 50

# Memory-constrained environment
python train.py --model_type lstm \
    --batch_size 64 \
    --epochs 100
```

### Performance Monitoring
```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# GPU usage monitoring
nvidia-smi -l 1

# View training logs
tail -f logs/train_*.log
```

## Troubleshooting

### Environment Issues
```bash
# Dependency installation
pip install -r requirements.txt

# HPO-specific dependencies (if running hyperparameter optimization)
pip install optuna

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
python -c "from models.model_factory import create_model; m=create_model('rwkv7', 40, 309); print('✓ RWKV-7 available')"

# vocab_size mismatch issues
python -c "from models.core import get_vocab_size; print(f'Current vocab_size: {get_vocab_size()}')"
```

### Common Error Solutions
1. **Out of Memory**: Reduce batch_size, use efficient model versions
2. **CUDA Errors**: Confirm GPU drivers, check torch.cuda.is_available()
3. **Model Loading Failure**: Check vocab_size consistency with training (26 vs 40)
4. **Accuracy Anomalies**: Verify dataset selection (S2/S4), check patterns.csv path
5. **Import Errors**: Confirm complete MODEL_REGISTRY in models/__init__.py
6. **HPO Training Failure**:
   - Ensure optuna is installed: `pip install optuna`
   - Check model parameter mapping is correct (fixed d_model mapping issue)
   - Verify max_epochs_per_trial setting in training configuration

## Key Considerations

1. **Dataset Consistency**: Ensure training, validation, testing use the same dataset (S2 or S4)
2. **vocab_size Matching**: Traditional models use 26, MDLM uses 40
3. **patterns.csv Correspondence**: Data post-processing uses correct patterns.csv file
4. **Directory Structure**: Model files saved in outputs/, code in models/, logs in logs/, TensorBoard in runs/
5. **Memory Management**: Adjust batch_size and model version selection based on GPU memory

## Quick Debug Commands

### One-Click Verification
```bash
# Complete environment check
python -c "
import torch
from models import MODEL_REGISTRY, list_available_models
from models.data_utils import create_data_loaders
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ Model count: {len(MODEL_REGISTRY)}')
print(f'✓ Supported models: {list_available_models()}')
try:
    train_loader, _, _ = create_data_loaders()
    print('✓ Data loading normal')
except Exception as e:
    print(f'✗ Data loading error: {e}')
"

# Quick model testing
python -c "
from models.model_factory import create_model
models_to_test = ['transformer', 'rwkv7', 'mdlm']
for model_type in models_to_test:
    try:
        vocab_size = 40 if model_type == 'mdlm' else 26
        model = create_model(model_type, vocab_size, 309)
        print(f'✓ {model_type}: {sum(p.numel() for p in model.parameters()):,} parameters')
    except Exception as e:
        print(f'✗ {model_type}: {e}')
"
```

### HPO Hyperparameter Optimization
```bash
# Run hyperparameter optimization experiments
python train_hpo.py --models rwkv7,mamba,transformer --n_trials 50

# Check HPO results
python -c "
import json
import glob
hpo_results = glob.glob('outputs/model_comparison_hpo_*/best_params.json')
for result_file in sorted(hpo_results):
    with open(result_file) as f:
        data = json.load(f)
        print(f'{result_file}: Best accuracy: {data.get(\"best_accuracy\", \"N/A\")}')
"
```

### Data Post-processing Pipeline Verification
```bash
# Complete data post-processing pipeline execution order
python data/decompose/01_out_original_reducer.py --input outputs/latest_output.out.original
python data/decompose/03_convert_reduced_to_final.py --input outputs/latest_output.out.reduced
python data/decompose/04_convert_final_to_out.py --input outputs/latest_output.out.final

# Verify accuracy calculation
python data/accuracy_analysis_system/accuracy_analyzer.py --output_file outputs/latest_output.out

# Other analysis tools
python data/compare_point_correctness/compare_final_units.py
python data/compare_word_level_correctness/verify_accuracy_simple.py
```

## Model Inference Examples

### Quick Inference Testing
```bash
# MDLM model inference (if available)
python -c "
import torch
from models.model_factory import create_model
from models.data_utils import CHAR_TO_IDX, IDX_TO_CHAR

# Create model and test input
model = create_model('mdlm', vocab_size=40, num_classes=309)
test_text = 'ܡܪܝܐ'
input_ids = [CHAR_TO_IDX.get(c, 0) for c in test_text]
input_tensor = torch.tensor([input_ids])

# Inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    predictions = output.argmax(-1)
    print(f'Input: {test_text}')
    print(f'Predictions: {predictions.tolist()}')
"
```

## Important Notes

### Dataset Selection
- **S2 Dataset** (default): Training set 50,649 lines, 309 classes, uses `data/patterns.csv`
- **S4 Dataset** (extended): Training set 97,048 lines, ~350 classes, uses `data/data_comparison/patterns.csv`
- When switching datasets, ensure correct patterns.csv file path

### Vocab Size Matching
- **Traditional Models** (transformer, bert, lstm): 26 characters
- **MDLM Models**: 40 characters (including digits and special tokens)
- **Model loading failures are usually caused by vocab_size mismatches**

### Performance Benchmarks
- **Current Baseline**: ~84% (LSTM/Transformer)
- **Target Performance**: 95%+ (cutting-edge architectures)
- **Best Expected**: 98%+ (ensemble models)

### Memory Management
Adjust batch_size based on GPU memory:
- 8GB GPU: batch_size=16-32
- 16GB GPU: batch_size=32-64
- 24GB+ GPU: batch_size=64+