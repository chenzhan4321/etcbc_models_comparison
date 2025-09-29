"""
Core utility module
Contains logging, device management and other basic functions
"""

import torch
import sys
from typing import Optional
from functools import wraps


class ConfigError(Exception):
    """Configuration error exception"""
    pass


class DataLoadError(Exception):
    """Data loading error exception"""
    pass


class ModelError(Exception):
    """Model related error exception"""
    pass


def log_info(message: str):
    """Output info log"""
    print(f"ℹ️ {message}", file=sys.stdout)


def log_warning(message: str):
    """Output warning log"""
    print(f"⚠️ {message}", file=sys.stderr)


def log_error(message: str):
    """Output error log"""
    print(f"❌ {message}", file=sys.stderr)


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_vocab_size(model_type: Optional[str] = None) -> int:
    """
    Get input vocabulary size (number of input characters)
    Note: MDLM model will calculate unified vocabulary size internally (input chars + label classes + special tokens)
    """
    if model_type and model_type.lower() == 'mdlm':
        return 40  # MDLM input character vocabulary (includes digits and special tokens)
    return 26  # Traditional models use basic Syriac character set


def print_device_info():
    """Print device information"""
    device = get_device()
    print(f"🔧 Using device: {device}")

    if device.type == 'cuda':
        print(f"   GPU name: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    elif device.type == 'mps':
        print(f"   Apple MPS enabled")
    else:
        print(f"   CPU mode")


def get_device_manager():
    """Get device manager (simplified version)"""
    class SimpleDeviceManager:
        def __init__(self):
            self.device = get_device()

        def get_device(self):
            return self.device

        def print_device_info(self):
            print_device_info()

    return SimpleDeviceManager()


def error_handler_decorator(func):
    """Error handling decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(f"Error occurred when executing {func.__name__}: {str(e)}")
            raise
    return wrapper


def create_optimized_config(model_type: str, batch_size: int = 32):
    """Create optimized configuration"""
    device = get_device()

    # Base configuration
    config = {
        'device': device,
        'batch_size': min(batch_size, 32),  # Limit batch size
        'model_config': {}
    }

    # Model-specific configuration
    if model_type.lower() in ['lstm', 'bilstm']:
        config['model_config'] = {
            'hidden_size': 256,
            'num_layers': 2,
            'max_length': 128
        }
    elif model_type.lower() in ['mdlm']:
        config['model_config'] = {
            'd_model': 256,
            'num_layers': 4,
            'num_heads': 8,
            'max_length': 128
        }
    else:  # transformer, bert, etc.
        config['model_config'] = {
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'max_length': 128
        }

    return config