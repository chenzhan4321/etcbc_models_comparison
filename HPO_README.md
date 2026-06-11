# Hyperparameter Optimization (HPO) Component User Guide

> **Scope note.** This is a general-purpose HPO utility that the codebase supports
> for many architectures. The **paper** only tunes and reports the encoder-only
> classifier, the MDLM, and the (matched) encoder–decoder baseline — see
> `README.md` and `hpo_seq2seq.py`. The other architectures listed below are
> exploratory and are **not part of the study**.

Hyperparameter tuning and model comparison system supporting all 13 neural network architectures.

## 🎯 Features

- **13 Model Support**: Transformer, BERT, LSTM, MDLM, Mamba, BiMamba, RWKV-7, RetNet, Switch, etc.
- **Intelligent Model Grouping**: Automatic grouping comparison by computational power/parameter count (light/medium/large/xl/efficient)
- **Multiple Optimization Algorithms**: Optuna (Bayesian), Hyperopt (TPE), Random Search, Grid Search
- **Smart Search Spaces**: Parameter spaces optimized for different architectures
- **Model Comparison Mode**: Two comparison modes: quick evaluation and deep HPO
- **Early Stopping Mechanism**: Automatic pruning of poorly performing trials
- **Resource Management**: Memory limits, training time control
- **Experiment Tracking**: Complete trial history and analysis reports
- **Multi-objective Support**: Accuracy, F1 score, loss function

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Basic dependencies
pip install torch numpy

# HPO algorithm libraries
pip install optuna  # Recommended
pip install hyperopt  # Optional

# Optional: Experiment tracking
pip install wandb
```

### 2. Basic Usage

#### Single Model HPO Optimization
```bash
# Transformer model HPO
python train_hpo.py \
    --model_type transformer \
    --optimizer optuna \
    --n_trials 50 \
    --max_epochs 20

# MDLM model HPO
python train_hpo.py \
    --model_type mdlm \
    --optimizer optuna \
    --n_trials 30 \
    --max_epochs 15 \
    --enable_pruning

# Newly supported models
python train_hpo.py --model_type bimamba_large --n_trials 40
python train_hpo.py --model_type rwkv7 --optimizer hyperopt --n_trials 35
python train_hpo.py --model_type retnet --n_trials 45 --enable_pruning
```

#### 🆕 Model Comparison Mode
```bash
# Quick comparison of all model groups
python train_hpo.py --compare_models --compare_mode quick --n_trials 5

# Compare specific model groups
python train_hpo.py --compare_models \
    --compare_groups light medium \
    --compare_mode hpo \
    --n_trials 20

# Deep comparison of large models
python train_hpo.py --compare_models \
    --compare_groups large xl \
    --compare_mode hpo \
    --n_trials 30 \
    --enable_pruning
```

## 📊 Model Grouping System

The HPO system automatically groups the 13 models by computational power/parameter count for equivalent-level comparison:

| Group | Models | Parameter Count | Description |
|-------|--------|----------------|-------------|
| **light** | lstm, transformer, bert | <10M | Lightweight, suitable for resource-constrained environments |
| **medium** | mdlm, mamba, bimamba, retnet | 10-20M | Medium scale, balancing performance and efficiency |
| **large** | mamba_large, bimamba_large, rwkv7, switch | 20-50M | Large models, high-performance requirements |
| **xl** | bimamba_xl, rwkv7_large | >50M | Extra-large models, ultimate performance |
| **efficient** | rwkv7_efficient, lstm | 2-15M | Optimized architectures, production-environment friendly |

## 📊 Detailed Parameters

### Basic Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model_type` | Model type | None | All 13 models |
| `--compare_models` | Enable model comparison | False | - |
| `--compare_groups` | Model groups to compare | all | light, medium, large, xl, efficient |
| `--compare_mode` | Comparison mode | hpo | hpo, quick |
| `--optimizer` | Optimization algorithm | optuna | optuna, hyperopt, random, grid |
| `--n_trials` | Number of trials | 50 | Any positive integer |
| `--timeout` | Time limit (seconds) | None | Any positive integer |
| `--max_epochs` | Maximum epochs per trial | 15 | 1-100 |

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--sampler` | Optuna sampler | tpe |
| `--enable_pruning` | Enable early stopping | False |
| `--memory_limit` | Memory limit (GB) | None |
| `--output_dir` | Output directory | auto |
| `--study_name` | Study name | auto |

### Search Space Customization

```bash
# Customize model dimensions
python train_hpo.py \
    --model_type transformer \
    --d_model_range "384,512,640" \
    --num_layers_range "4,6,8,10"

# Fixed batch size
python train_hpo.py \
    --model_type mdlm \
    --batch_size_override 32
```

## 🔧 Programming Interface

### Basic Usage

```python
from models.components.hpo import (
    HPOConfig, HPOSearchSpace, create_hpo_optimizer
)

# Configure HPO
config = HPOConfig(
    study_name="my_experiment",
    n_trials=50,
    optimizer="optuna",
    direction="maximize",
    metric="val_accuracy"
)

# Define search space
search_space = HPOSearchSpace(
    model_type="transformer",
    d_model=[256, 384, 512],
    num_layers=[4, 6, 8],
    learning_rate=(1e-5, 1e-3)
)

# Create optimizer
optimizer = create_hpo_optimizer(
    model_type="transformer",
    train_func=your_train_function,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="outputs/hpo",
    config=config,
    search_space=search_space
)

# Start optimization
results = optimizer.optimize()
```

### Custom Training Function

```python
def your_train_function(model_type, params, train_loader, val_loader,
                       max_epochs=20, trial_id=0, trial=None):
    """
    Custom training function

    Args:
        model_type: Model type
        params: Hyperparameter dictionary
        train_loader: Training data
        val_loader: Validation data
        max_epochs: Maximum training epochs
        trial_id: Trial ID
        trial: Optuna trial object (for pruning)

    Returns:
        dict: Contains val_accuracy, val_f1, val_loss and other metrics
    """
    # Create model
    model = create_model(model_type, params)

    # Training loop
    for epoch in range(max_epochs):
        # Training code...
        val_metrics = validate_model(model, val_loader)

        # Optuna pruning support
        if trial is not None:
            trial.report(val_metrics['accuracy'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return val_metrics
```

## 📈 Search Space Details

### Transformer Search Space

```python
search_space = HPOSearchSpace(
    model_type="transformer",
    # Architecture parameters
    d_model=[256, 384, 512, 640, 768],      # Model dimension
    num_layers=[3, 4, 6, 8, 10],            # Number of layers
    num_heads=[4, 6, 8, 12, 16],            # Number of attention heads
    dim_feedforward=[1024, 1536, 2048, 3072], # FFN dimension

    # Training parameters
    dropout=(0.0, 0.5),                      # Dropout rate
    learning_rate=(1e-5, 1e-3),             # Learning rate
    batch_size=[16, 32, 64],                 # Batch size
    weight_decay=(0.0, 0.1),                # Weight decay

    # Other parameters
    activation=["relu", "gelu", "swish"],    # Activation function
    layer_norm_eps=(1e-6, 1e-4)             # LayerNorm epsilon
)
```

### MDLM Search Space

```python
search_space = HPOSearchSpace(
    model_type="mdlm",
    # Architecture parameters
    d_model=[256, 384, 512],                 # Model dimension
    num_layers=[3, 4, 6, 8],                # Number of layers

    # MDLM-specific parameters
    num_timesteps=[500, 1000, 2000],         # Diffusion timesteps
    mask_ratio=(0.1, 0.5),                  # Masking ratio
    diffusion_steps=[5, 10, 20],            # Diffusion steps
    max_length=[64, 128, 256],              # Maximum length

    # Training parameters
    dropout=(0.0, 0.3),
    learning_rate=(1e-5, 5e-4),
    batch_size=[16, 32, 64],
    weight_decay=(0.0, 0.05)
)
```

## 📊 Results Analysis

### Viewing Optimization Results

After training completion, results are saved in the output directory:

```
outputs/hpo_<model>_<timestamp>/
├── hpo.log                          # Training log
├── hpo_config.json                  # HPO configuration
├── trial_history.json               # All trial history
├── best_result.json                 # Best results
├── optimization_results.json        # Optimization results summary
├── hpo_analysis.json               # Results analysis
└── recommended_command.txt          # Recommended training command
```

### Results Analysis API

```python
from models.components.hpo import analyze_hpo_results

# Analyze results
analysis = analyze_hpo_results("outputs/hpo_transformer_20240101_120000")

print(f"Best score: {analysis['best_score']}")
print(f"Parameter importance: {analysis['parameter_importance']}")
print(f"Score distribution: {analysis['score_distribution']}")
```

### Visualizing Results

```python
import matplotlib.pyplot as plt
import json

# Read trial history
with open('outputs/hpo_results/trial_history.json') as f:
    trials = json.load(f)

# Plot optimization curve
scores = [trial['score'] for trial in trials]
plt.figure(figsize=(10, 6))
plt.plot(scores)
plt.xlabel('Trial')
plt.ylabel('Validation Accuracy')
plt.title('HPO Optimization Progress')
plt.show()

# Plot parameter importance
analysis = analyze_hpo_results('outputs/hpo_results')
importance = analysis['parameter_importance']

plt.figure(figsize=(10, 6))
params = list(importance.keys())[:10]  # Top 10
values = list(importance.values())[:10]
plt.barh(params, values)
plt.xlabel('Importance')
plt.title('Parameter Importance')
plt.tight_layout()
plt.show()
```

## 💡 Best Practices

### 1. Trial Count Selection

| Model Complexity | Recommended Trials | Description |
|------------------|-------------------|-------------|
| Simple Models | 20-50 | Quick validation |
| Medium Models | 50-100 | Balance effectiveness and time |
| Complex Models | 100-200 | Thorough exploration |

### 2. Search Space Design

```python
# ✅ Good search space - moderate range
search_space = HPOSearchSpace(
    d_model=[256, 384, 512],      # 3 choices
    num_layers=[4, 6, 8],         # 3 choices
    learning_rate=(1e-5, 1e-3)    # Reasonable range
)

# ❌ Poor search space - oversized range
search_space = HPOSearchSpace(
    d_model=list(range(64, 1024, 32)),  # Too many choices
    num_layers=list(range(1, 50)),      # Range too large
    learning_rate=(1e-8, 1e-1)          # Range too large
)
```

### 3. Resource Management

```bash
# Limit training time per trial
python train_hpo.py \
    --model_type transformer \
    --max_epochs 15 \
    --timeout 3600  # 1-hour total time limit

# Enable early stopping to save resources
python train_hpo.py \
    --model_type mdlm \
    --enable_pruning \
    --memory_limit 8  # 8GB memory limit
```

### 4. Multi-stage Optimization

```bash
# Stage 1: Coarse search
python train_hpo.py \
    --model_type transformer \
    --n_trials 30 \
    --max_epochs 10 \
    --d_model_range "256,384,512,640"

# Stage 2: Fine search (based on stage 1 results)
python train_hpo.py \
    --model_type transformer \
    --n_trials 50 \
    --max_epochs 20 \
    --d_model_range "384,512"  # Narrowed range
```

## ⚡ Performance Optimization

### 1. Parallel Trials

```python
# Multi-process parallel (requires sufficient GPU memory)
config = HPOConfig(
    n_jobs=2,  # 2 parallel trials
    n_trials=100
)
```

### 2. Early Termination

```python
config = HPOConfig(
    enable_pruning=True,
    pruning_patience=5,        # Stop after 5 rounds without improvement
    min_trials_for_pruning=10  # Start pruning after at least 10 trials
)
```

### 3. Caching and Reuse

```bash
# Use fixed study name to reuse historical results
python train_hpo.py \
    --study_name "transformer_hpo_v1" \
    --n_trials 50  # Will continue previous optimization
```

## 🐛 Common Issues

### Q: Running out of memory?
A: Use `--memory_limit` parameter, reduce `batch_size` search range, enable pruning

### Q: Optimization taking too long?
A: Reduce `--max_epochs`, use `--timeout` parameter, enable early stopping

### Q: Results unstable?
A: Increase trial count, fix random seed, average multiple runs

### Q: How to choose optimization algorithm?
A:
- Small search space: random search
- Medium search space: optuna (TPE)
- Large search space: optuna (CMA-ES)
- Want complete exploration: grid search

## 📝 Practical Usage Examples

### Finding Optimal Configuration for Transformer

```bash
# Step 1: Quick exploration
python train_hpo.py \
    --model_type transformer \
    --optimizer optuna \
    --n_trials 50 \
    --max_epochs 10 \
    --enable_pruning \
    --output_dir outputs/transformer_hpo_phase1

# Step 2: Fine-tuning based on results
# Check outputs/transformer_hpo_phase1/recommended_command.txt
# Then perform full training based on recommended parameters
```

### Finding Optimal Configuration for MDLM

```bash
# MDLM requires more careful parameter tuning
python train_hpo.py \
    --model_type mdlm \
    --optimizer optuna \
    --n_trials 30 \
    --max_epochs 15 \
    --d_model_range "256,384,512" \
    --num_layers_range "3,6,8" \
    --output_dir outputs/mdlm_hpo

# Check results and train with recommended parameters
cat outputs/mdlm_hpo/recommended_command.txt
```

## 📚 References

- [Optuna Documentation](https://optuna.org/)
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)
- [Bayesian Optimization Principles](https://distill.pub/2020/bayesian-optimization/)