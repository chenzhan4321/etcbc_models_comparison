# SyriacML: Neural Networks for Syriac Morphological Analysis

![Version](https://img.shields.io/badge/version-12.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Open%20Source-green.svg)

##  Project Overview

Deep learning project for Syriac morphological analysis supporting 13 neural network architectures. **Core task**: sequence labeling — given Syriac character sequences, predict morphological tags for each position.

### Key Features
- **2 Core Architectures**: Transformer (Encoder-only), MDLM (Masked Diffusion)
- **10 Experimental Architectures**: BERT, Mamba, BiMamba, RetNet, Switch, RWKV-7 (3 variants) for research
- **Dual Dataset Support**: S2 (default) and S4 (extended) datasets with consistent test sets
- **Production-Ready Pipeline**: Complete training, validation, evaluation, and deployment system
- **Reproducible Results**: Multi-seed experiments with detailed statistical analysis

##  Dataset System

The project supports two dataset configurations with **strict separation** for fair comparison:

| Configuration | Training Data | Test Data | Training Size | Path |
|---------------|---------------|-----------|---------------|------|
| **S2-on-S2** (default) | S2 | S2 | 50,649 lines | `data/s2_on_s2/` |
| **S4-on-S2** (extended) | S4 | S2 | 97,048 lines | `data/s4_on_s2/` |

### Dataset Details
- **S2-on-S2**: Standard configuration using S2 dataset for both training and testing
- **S4-on-S2**: Extended training with S4 data (1.92× more samples), evaluated on consistent S2 test set
- **Validation & Test**: Both configurations use identical validation (10,964) and test (10,869) sets from S2
- **Classes**: S2 has 309 classes; S4 training includes ~350 classes (40 additional classes for extended coverage)

##  Performance Results

All models evaluated on the S2 test set (10,869 lines) using **5-seed reproducibility experiments**.

### Average Edit Distance (Levenshtein) - Lower is Better

| Model | S4-on-S2 (Extended Training) | S2-on-S2 (Standard Training) | Improvement |
|-------|------------------------------|------------------------------|-------------|
| **MDLM** (d=768, L=5, H=4) | **3.42% ± 0.08%** | 3.64% ± 0.05% | **-0.22%** |
| **Encoder-only Transformer** (d=512, L=4, H=16) | **3.53% ± 0.04%** | 3.88% ± 0.03% | **-0.35%** |
| **Encoder-Decoder Transformer** (d=512, H=8) | ~4.08% | ~4.00% | **-0.20%** |
|                                                 |                              |                              |             |

### Key Findings
- ** Extended training (S4-on-S2) consistently improves performance** across all models by 0.2-0.35% in CER
- ** MDLM achieves best character-level accuracy** (96.58%) and lowest CER (3.42%) on S4-on-S2
- ** Encoder-only Transformer offers best balance** of performance (3.53% CER) and efficiency
- ** High reproducibility** across seeds: CV < 0.1% for all core models (EXCELLENT rating)
- ** Line-level accuracy shows larger gains** from extended training (+2.5-3.2 percentage points)

##  Evaluation Reports

Detailed evaluation results are available in the `report/` directory:

```
report/
├── encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/
│   ├── s2_on_s2/
│   │   ├── levenshtein_summary.txt                  # Character: 96.12% ± 0.03%, CER: 3.88%
│   │   └── transformer_train_TIMESTAMP_seed*/       # Individual seed results
│   └── s4_on_s2/
│       ├── levenshtein_summary.txt                  # Character: 96.47% ± 0.04%, CER: 3.53%
│       └── transformer_train_TIMESTAMP_seed*/       # Individual seed results
│
├── mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds/
│   ├── s2-on-s2/
│   │   ├── levenshtein_summary_correct.txt          # Character: 96.36% ± 0.05%, CER: 3.64%
│   │   └── mdlm_train_TIMESTAMP_seed*/              # Individual seed results
│   └── s4-on-s2/
│       ├── levenshtein_summary_correct.txt          # Character: 96.58% ± 0.08%, CER: 3.42%
│       └── mdlm_train_TIMESTAMP_seed*/              # Individual seed results
│
├── encoder_decoder_30ep_bs128_lr1e-4_emb512_h8_d0.1_b3_s7/
│   ├── s2_on_s2/
│   │   └── results/                                 # Character: ~96.00%, CER: ~4.00%
│   └── s4_on_s4 (not included)/
│       └── results/                                 # Character: ~96.20%, CER: ~3.80%
│
└── case_study_mdlm_50ep_bs16_lr3e-4_d384_l6_h4_d0.25_steps2_s52/
    ├── results/
    │   ├── levenshtein_correct.txt                  # Single-seed case study
    │   ├── token_vs_char_accuracy_analysis.md       # Token-level vs character-level analysis
    │   └── levenshtein_detailed_analysis.md         # Detailed error analysis
    └── data/patterns.csv
```

### Report Structure
- **levenshtein_summary.txt / levenshtein_summary_correct.txt**: Aggregated statistics across all seeds
  - Character-level accuracy and CER
  - Line-level accuracy (exact match rate)
  - Average Levenshtein edit distance
  - Per-seed breakdown with mean and standard deviation
  - Reproducibility assessment (Coefficient of Variation)
- **Individual seed directories**: Complete results for each random seed
  - Model outputs (.out.original, .out.reduced, .out.final, .out)
  - Levenshtein distance calculations
  - Training history and checkpoints
- **Case studies**: In-depth analysis of specific model configurations

##  Project Architecture

### Directory Structure
```
etcbc_models/
├── models/                      # Model definitions (pure Python code)
│   ├── core/                    # Core utilities (device, logging, vocabulary)
│   ├── components/              # Reusable neural network components
│   ├── base.py                  # Base model classes
│   ├── mdlm.py                  # MDLM (Masked Diffusion) implementation
│   ├── cutting_edge.py          # RWKV-7, Mamba, RetNet implementations
│   └── model_factory.py         # Model factory for dynamic instantiation
├── outputs/                     # Training results and model checkpoints
├── report/                      # Evaluation reports and analysis
├── data/                        # Training data and preprocessing scripts
├── data_processing_tools/       # Data analysis and processing utilities
├── reproducibility/             # Multi-seed reproducibility experiments
├── runs/                        # TensorBoard logs (excluded from git)
└── logs/                        # Training logs (excluded from git)
```

### Core Modules
- **`train.py`**: Main training script supporting all 13 architectures
- **`train_hpo.py`**: Hyperparameter optimization (HPO) with Optuna
- **`inference.py`**: Production inference script with CLI
- **`models/`**: Model implementation package with MODEL_REGISTRY
- **`models/data_utils.py`**: Data loading and preprocessing (default: S2 dataset)
- **`models/model_factory.py`**: Factory pattern for model instantiation
- **`models/config_manager.py`**: Centralized configuration management
- **`data_processing_tools/`**: Data post-processing and accuracy analysis

##  Core Model Architectures

The project focuses on **three primary architectures**, each representing different paradigms:

###  Transformer (Encoder-Only)
**Standard attention-based sequence labeling model**
- **Architecture**: Multi-head self-attention with sinusoidal positional encoding
- **Complexity**: O(n²) time and memory
- **Vocabulary**: 26 characters (basic Syriac character set: 1 space + 25 letters)
- **Configuration**: d=512, L=4, H=16, dropout=0.25
- **Key Features**:
  - Parallel processing of entire sequences
  - Global attention mechanism captures long-range dependencies
  - Strong baseline performance with good efficiency
- **Performance**: 3.53% CER (S4-on-S2), 3.88% CER (S2-on-S2)
- **Use Cases**: Production baseline, standard benchmark, resource-constrained deployments
- **Training**:
  ```bash
  python train.py --model_type transformer --epochs 100 --batch_size 128 \
      --d_model 512 --num_layers 4 --num_heads 16 --dropout 0.25
  ```

###  MDLM (Masked Diffusion Language Model)
**Discrete diffusion model for iterative sequence refinement**
- **Architecture**: Transformer-based diffusion with discrete masking and denoising
- **Complexity**: O(n²) time with T diffusion steps (T=2 in production)
- **Vocabulary**: 40 characters (extended set: 26 basic + 10 digits + 4 special tokens)
- **Configuration**: d=768, L=5, H=4, dropout=0.25, timesteps=2
- **Key Features**:
  - Iterative refinement through forward diffusion and reverse denoising
  - Handles morphological ambiguity and uncertainty naturally
  - Generation capabilities for data augmentation
  - Discrete masking with learnable noise schedule
- **Performance**: 3.42% CER (S4-on-S2), 3.64% CER (S2-on-S2) — **Best character accuracy**
- **Use Cases**: High-quality morphological analysis, uncertainty quantification, research
- **Training**:
  ```bash
  python train.py --model_type mdlm --epochs 200 --batch_size 16 \
      --d_model 768 --num_layers 5 --num_heads 4 --dropout 0.25 --num_timesteps 2
  ```

##  Experimental Architectures

Additional architectures available for research and experimentation:

| Architecture | Key Innovation | Time Complexity | Status |
|--------------|----------------|-----------------|--------|
| **LSTM** | Long Short-Term Memory | O(n) | Experimental |
| **Mamba** | State-space models with selective scanning | O(n) | Experimental |
| **BiMamba** | Bidirectional Mamba | O(n) | Experimental |
| **RetNet** | Retention mechanism for parallel training | O(n) | Experimental |
| **Switch** | Mixture of Experts (MoE) | O(n²) | Experimental |
| **RWKV-7** | Linear attention with time-mixing | O(n) | Experimental |
| **RWKV-7 Large** | Scaled-up RWKV-7 | O(n) | Experimental |
| **RWKV-7 Efficient** | Memory-optimized RWKV-7 | O(n) | Experimental |

**Quick Start with Experimental Models:**

```bash
python train.py --model_type bert --epochs 30 --batch_size 32
python train.py --model_type mamba --epochs 50 --batch_size 32
python train.py --model_type rwkv7 --epochs 50 --batch_size 16
python train.py --model_type retnet --epochs 60 --batch_size 32
```

##  Data Processing System

### Character Encoding
- **Basic Set**: 26 Syriac characters (1 space + 25 letters)
  - Used by: Transformer (encoder-only), LSTM, experimental models
- **Extended Set**: 40 characters (26 basic + 10 digits + 4 special tokens)
  - Used by: MDLM (requires extended vocabulary for masking)
  - Special tokens: `[MASK]`, `[PAD]`, `[UNK]`, `[CLS]`

### Data Post-processing Pipeline
```
Model Raw Output (.out.original)
    ↓ [Simplification]
    data_processing_tools/decompose/01_out_original_reducer.py
    ↓
Reduced Format (.out.reduced)
    ↓ [Transformation]
    data_processing_tools/decompose/03_convert_reduced_to_final.py
    ↓
Final Format (.out.final)
    ↓ [Standardization]
    data_processing_tools/decompose/04_convert_final_to_out.py
    ↓
Standard Output (.out)  ← Used for accuracy evaluation
```

### Evaluation Metrics

#### Character Error Rate (CER) = Average Edit Distance
```
CER = (Substitutions + Insertions + Deletions) / Total_Characters × 100%
```
Primary metric for model comparison. Lower is better.

### Data Directory Structure
```
data/
├── s2_on_s2/                    # Default configuration
│   ├── train.in                 # Training inputs (50,649 lines)
│   ├── train.out                # Training labels
│   ├── val.in                   # Validation inputs (10,964 lines)
│   ├── val.out                  # Validation labels
│   ├── test.in                  # Test inputs (10,869 lines)
│   ├── test.out                 # Test labels
│   └── patterns.csv             # 309 morphological patterns
├── s4_on_s2/                    # Extended training configuration
│   ├── train.in                 # Training inputs (97,048 lines)
│   ├── train.out                # Training labels
│   ├── val.in                   # Same as S2 (10,964 lines)
│   ├── val.out                  # Same as S2
│   ├── test.in                  # Same as S2 (10,869 lines)
│   ├── test.out                 # Same as S2
│   └── patterns.csv             # ~350 morphological patterns
└── data_processing_tools/
    ├── decompose/               # Output format conversion
    ├── accuracy_analysis_system/ # Accuracy computation
    ├── levenshtein/             # Edit distance evaluation
    └── compare_word_level_correctness/ # Word-level metrics
```

##  Key Technical Details

### Vocab Size Matching
- **Transformer (encoder-only)**: 26 characters
- **LSTM**: 26 characters
- **MDLM**: 40 characters (extended vocabulary required for masking)
- **Experimental models**: Most use 26; check model documentation

### Dataset Selection
- Always use **S2 test set** for evaluation (never S4 test set)
- S4-on-S2 trains on S4 but tests on S2 for fair comparison
- patterns.csv must match the dataset: s2_on_s2 uses 309 patterns, s4_on_s2 uses ~350

### Directory Naming Convention
```
{model}_{epochs}ep_bs{batch_size}_lr{learning_rate}_d{d_model}_l{num_layers}_h{num_heads}_d{dropout}[_steps{num_timesteps}]_{num_seeds}seeds
```

Examples:
- `encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds`
- `mdlm_200ep_bs16_lr1e-4_d768_l5_h4_d0.25_steps2_5seeds`

**Note**: Encoder-only does not use diffusion steps; MDLM uses num_timesteps (typically 2 or 3 for efficiency).

##  License

This project follows open source licensing requirements. See LICENSE file for details.

##  Acknowledgments

- Syriac morphological data from ETCBC (Eep Talstra Centre for Bible and Computer)

##  Contact

For questions, issues, or collaboration inquiries, please open an issue on the repository.

---

**Last Updated**: 2026-01-02
**Version**: 12.0.0
**Status**: Production-ready core models, experimental research models available
