"""
核心工具模块
包含日志、设备管理等基础功能
支持: CUDA (NVIDIA), MPS (Apple Silicon), 多核CPU
"""

import torch
import sys
import os
import multiprocessing
from typing import Optional, Dict, Any
from functools import wraps
from dataclasses import dataclass


class ConfigError(Exception):
    """配置错误异常"""
    pass


class DataLoadError(Exception):
    """数据加载错误异常"""
    pass


class ModelError(Exception):
    """模型相关错误异常"""
    pass


def log_info(message: str):
    """输出信息日志"""
    print(f"ℹ️ {message}", file=sys.stdout)


def log_warning(message: str, category: Optional[str] = None):
    """输出警告日志"""
    prefix = f"{category}: " if category else ""
    print(f"⚠️ {prefix}{message}", file=sys.stderr)


def log_error(message: str):
    """输出错误日志"""
    print(f"❌ {message}", file=sys.stderr)


@dataclass
class DeviceConfig:
    """设备配置"""
    device: torch.device
    device_type: str  # 'cuda', 'mps', 'cpu'
    num_workers: int  # DataLoader workers
    pin_memory: bool  # 是否锁定内存
    batch_size: int   # 推荐批次大小
    num_threads: int  # CPU线程数
    compile_model: bool  # 是否使用torch.compile
    memory_gb: float  # 可用内存(GB)


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    info = {
        'cpu_count': multiprocessing.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'pytorch_version': torch.__version__,
    }

    # 获取内存信息
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_total_gb'] = mem.total / (1024**3)
        info['memory_available_gb'] = mem.available / (1024**3)
    except ImportError:
        # 如果没有psutil，使用保守估计
        info['memory_total_gb'] = 8.0
        info['memory_available_gb'] = 4.0

    # CUDA信息
    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['cuda_total_memory_gb'] = info['cuda_memory_gb'] * info['cuda_device_count']

    return info


def setup_multi_gpu(model: torch.nn.Module, device_ids: list = None) -> torch.nn.Module:
    """
    设置多GPU并行训练

    Args:
        model: PyTorch模型
        device_ids: GPU ID列表，如 [0, 1, 2, 3]，None表示使用所有可用GPU

    Returns:
        包装后的并行模型
    """
    if not torch.cuda.is_available():
        log_warning("CUDA不可用，无法使用多GPU")
        return model

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        log_info(f"只有{num_gpus}个GPU，使用单GPU模式")
        return model.cuda()

    if device_ids is None:
        device_ids = list(range(num_gpus))

    log_info(f"🚀 启用多GPU并行训练: {len(device_ids)} GPUs {device_ids}")

    # 使用DataParallel包装模型
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    return model


def get_multi_gpu_batch_size(base_batch_size: int, num_gpus: int = None) -> int:
    """
    计算多GPU的有效批次大小

    Args:
        base_batch_size: 单GPU批次大小
        num_gpus: GPU数量，None表示自动检测

    Returns:
        总批次大小 = base_batch_size * num_gpus
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    return base_batch_size * num_gpus


def get_device() -> torch.device:
    """获取最佳可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def configure_cpu_optimization(num_threads: Optional[int] = None):
    """配置CPU优化参数"""
    cpu_count = multiprocessing.cpu_count()

    if num_threads is None:
        # 使用80%的CPU核心，留一些给系统
        num_threads = max(1, int(cpu_count * 0.8))

    # 设置PyTorch线程数
    torch.set_num_threads(num_threads)

    # 设置OpenMP线程数
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    # 设置MKL线程数（如果使用Intel MKL）
    os.environ['MKL_NUM_THREADS'] = str(num_threads)

    # 设置线程亲和性
    os.environ['OMP_PROC_BIND'] = 'close'
    os.environ['OMP_PLACES'] = 'cores'

    return num_threads


def get_optimal_device_config(
    force_device: Optional[str] = None,
    force_batch_size: Optional[int] = None
) -> DeviceConfig:
    """
    获取最优设备配置

    支持三种场景:
    1. CUDA (NVIDIA GPU) - 使用GPU加速
    2. MPS (Apple Silicon) - 使用Metal加速
    3. CPU (多核) - 使用多线程并行
    """
    sys_info = get_system_info()

    # 确定设备类型
    if force_device:
        device_type = force_device.lower()
        device = torch.device(device_type)
    elif sys_info['cuda_available']:
        device_type = 'cuda'
        device = torch.device('cuda')
    elif sys_info['mps_available']:
        device_type = 'mps'
        device = torch.device('mps')
    else:
        device_type = 'cpu'
        device = torch.device('cpu')

    # 根据设备类型配置参数
    if device_type == 'cuda':
        # CUDA配置
        gpu_mem = sys_info.get('cuda_memory_gb', 8.0)
        config = DeviceConfig(
            device=device,
            device_type='cuda',
            num_workers=4,
            pin_memory=True,
            batch_size=64 if gpu_mem >= 8 else 32,
            num_threads=4,  # CUDA不需要太多CPU线程
            compile_model=False,  # CUDA通常不需要compile
            memory_gb=gpu_mem,
        )

    elif device_type == 'mps':
        # Apple MPS配置
        config = DeviceConfig(
            device=device,
            device_type='mps',
            num_workers=0,  # MPS不支持多进程DataLoader
            pin_memory=False,  # MPS不需要pin_memory
            batch_size=16,  # MPS内存有限
            num_threads=4,
            compile_model=False,  # MPS暂不完全支持compile
            memory_gb=sys_info.get('memory_available_gb', 8.0),
        )

    else:
        # CPU多核配置
        cpu_count = sys_info['cpu_count']
        mem_gb = sys_info.get('memory_available_gb', 8.0)

        # 根据内存和CPU核心数计算最优配置
        num_threads = configure_cpu_optimization()

        # DataLoader workers: 使用一半的CPU核心
        num_workers = min(cpu_count // 2, 16)

        # 批次大小: 根据内存调整
        # 假设每个样本约1KB，每批次需要额外2x内存用于梯度
        if mem_gb >= 64:
            batch_size = 256
        elif mem_gb >= 32:
            batch_size = 128
        elif mem_gb >= 16:
            batch_size = 64
        else:
            batch_size = 32

        # PyTorch 2.0+ 支持 torch.compile
        can_compile = hasattr(torch, 'compile') and sys.version_info >= (3, 8)

        config = DeviceConfig(
            device=device,
            device_type='cpu',
            num_workers=num_workers,
            pin_memory=False,
            batch_size=batch_size,
            num_threads=num_threads,
            compile_model=can_compile,
            memory_gb=mem_gb,
        )

    # 允许覆盖批次大小
    if force_batch_size:
        config.batch_size = force_batch_size

    return config


def print_device_config(config: DeviceConfig):
    """打印设备配置信息"""
    print(f"\n{'='*50}")
    print(f"🔧 设备配置")
    print(f"{'='*50}")
    print(f"   设备类型: {config.device_type.upper()}")
    print(f"   设备: {config.device}")
    print(f"   可用内存: {config.memory_gb:.1f} GB")
    print(f"   批次大小: {config.batch_size}")
    print(f"   DataLoader workers: {config.num_workers}")
    print(f"   CPU线程数: {config.num_threads}")
    print(f"   Pin Memory: {config.pin_memory}")
    print(f"   torch.compile: {config.compile_model}")
    print(f"{'='*50}\n")


def get_vocab_size(model_type: Optional[str] = None) -> int:
    """
    获取输入词汇表大小（输入字符数量）
    注意：MDLM模型会在内部计算统一词汇表大小（输入字符 + 标签类别 + 特殊标记）
    """
    if model_type and model_type.lower() == 'mdlm':
        return 40  # MDLM模型使用40字符词汇表（包含数字和特殊标记）
    return 26  # 传统模型使用基础叙利亚文字符集


def print_device_info():
    """打印设备信息"""
    config = get_optimal_device_config()
    print_device_config(config)


class DeviceManager:
    """
    统一设备管理器
    支持 CUDA / MPS / 多核CPU
    """

    def __init__(self, force_device: Optional[str] = None, force_batch_size: Optional[int] = None):
        self.config = get_optimal_device_config(force_device, force_batch_size)
        self.device = self.config.device
        self._model_compiled = False

    def get_device(self) -> torch.device:
        return self.device

    def get_config(self) -> DeviceConfig:
        return self.config

    def print_device_info(self):
        print_device_config(self.config)

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """获取DataLoader的优化参数"""
        return {
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.num_workers > 0,
        }

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """应用设备特定的模型优化"""
        model = model.to(self.device)

        # CPU多核: 使用torch.compile加速
        if self.config.compile_model and not self._model_compiled:
            try:
                model = torch.compile(model, mode='reduce-overhead')
                log_info("已应用 torch.compile 优化")
                self._model_compiled = True
            except Exception as e:
                log_warning(f"torch.compile 失败: {e}")

        return model

    def get_recommended_batch_size(self) -> int:
        """获取推荐批次大小"""
        return self.config.batch_size

    def summary(self) -> str:
        """返回设备配置摘要"""
        return (
            f"{self.config.device_type.upper()} | "
            f"Batch={self.config.batch_size} | "
            f"Workers={self.config.num_workers} | "
            f"Threads={self.config.num_threads}"
        )


def get_device_manager(
    force_device: Optional[str] = None,
    force_batch_size: Optional[int] = None
) -> DeviceManager:
    """获取设备管理器"""
    return DeviceManager(force_device, force_batch_size)


def error_handler_decorator(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(f"执行 {func.__name__} 时发生错误: {str(e)}")
            raise
    return wrapper


def create_optimized_config(model_type: str, batch_size: int = 32):
    """创建优化配置"""
    device = get_device()

    # 基础配置
    config = {
        'device': device,
        'batch_size': min(batch_size, 32),  # 限制批大小
        'model_config': {}
    }

    # 模型特定配置
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
    else:  # transformer, bert等
        config['model_config'] = {
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'max_length': 128
        }

    return config