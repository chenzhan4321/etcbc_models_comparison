"""
配置管理模块
用于统一管理项目的配置参数和设置

这个模块的主要功能：
1. 统一的配置文件格式
2. 配置验证和默认值
3. 环境变量支持
4. 配置文件热重载
5. 不同模型的配置模板
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from models.core import ConfigError, log_info, log_warning


@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str = "transformer"
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_length: int = 64
    vocab_size: int = None  # 将在运行时确定
    num_classes: int = None  # 将在运行时确定


@dataclass
class TrainingConfig:
    """训练配置类"""
    batch_size: int = 16
    learning_rate: float = 2e-5  # 优化为适合BERT/Transformer的学习率
    num_epochs: int = 50
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    save_steps: int = 500
    eval_steps: int = 100
    early_stopping_patience: int = 7
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """数据配置类"""
    data_dir: str = "./data"
    train_input: str = "train.in"
    train_output: str = "train.out"
    val_input: str = "val.in"
    val_output: str = "val.out"
    test_input: str = "test.in"
    test_output: str = "test.out"
    max_length: int = 64
    min_length: int = 1
    cache_data: bool = True


@dataclass
class SystemConfig:
    """系统配置类"""
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_model: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    log_level: str = "INFO"
    save_dir: str = "./models"
    log_dir: str = "./logs"


@dataclass
class Config:
    """完整配置类"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    system: SystemConfig
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保数据路径是绝对路径
        if not os.path.isabs(self.data.data_dir):
            self.data.data_dir = os.path.abspath(self.data.data_dir)
        
        # 确保保存和日志目录存在
        os.makedirs(self.system.save_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)


class ConfigManager:
    """
    配置管理器
    
    负责加载、验证和管理配置文件
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        参数：
            config_file (str, optional): 配置文件路径
        """
        self.config_file = config_file
        self.config = None
        self._default_configs = self._create_default_configs()
        
    def _create_default_configs(self) -> Dict[str, Config]:
        """创建默认配置模板"""
        configs = {}
        
        # Transformer配置
        configs["transformer"] = Config(
            model=ModelConfig(
                model_type="transformer",
                d_model=256,
                num_layers=3,
                num_heads=4,
                dropout=0.1
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=0.0001,  # 降低学习率以防止模型过快收敛
                num_epochs=50
            ),
            data=DataConfig(),
            system=SystemConfig()
        )
        
        # BERT配置
        configs["bert"] = Config(
            model=ModelConfig(
                model_type="bert",
                d_model=256,
                num_layers=3,
                num_heads=4,
                dropout=0.1
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=0.0001,  # BERT通常使用较小的学习率
                num_epochs=30
            ),
            data=DataConfig(),
            system=SystemConfig()
        )
        
        # BiLSTM配置
        configs["bilstm"] = Config(
            model=ModelConfig(
                model_type="bilstm",
                d_model=256,  # BiLSTM通常使用较小的模型
                num_layers=2,
                dropout=0.2
            ),
            training=TrainingConfig(
                batch_size=32,  # BiLSTM可以使用较大的批次
                learning_rate=0.001,
                num_epochs=50
            ),
            data=DataConfig(),
            system=SystemConfig()
        )
        
        # 扩散模型配置
        configs["diffusion"] = Config(
            model=ModelConfig(
                model_type="diffusion",
                d_model=256,
                num_layers=4,
                num_heads=8,
                dropout=0.1,
                max_length=128  # 调整为128以适应最长的.final文件
            ),
            training=TrainingConfig(
                batch_size=8,  # 扩散模型内存需求较大
                learning_rate=0.0001,
                num_epochs=100  # 扩散模型需要更多训练轮数
            ),
            data=DataConfig(
                max_length=128  # 数据处理也使用128的长度限制
            ),
            system=SystemConfig()
        )
        
        return configs
    
    def load_config(self, config_file: Optional[str] = None, 
                   model_type: str = "transformer") -> Config:
        """
        加载配置
        
        参数：
            config_file (str, optional): 配置文件路径
            model_type (str): 模型类型，用于选择默认配置
            
        返回：
            Config: 配置对象
        """
        config_file = config_file or self.config_file
        
        if config_file and os.path.exists(config_file):
            # 从文件加载配置
            log_info(f"从文件加载配置: {config_file}", "配置管理")
            self.config = self._load_from_file(config_file)
        else:
            # 使用默认配置
            log_info(f"使用默认配置: {model_type}", "配置管理")
            if model_type in self._default_configs:
                self.config = self._default_configs[model_type]
            else:
                log_warning(f"未知模型类型: {model_type}，使用transformer配置", "配置管理")
                self.config = self._default_configs["transformer"]
                self.config.model.model_type = model_type
        
        # 验证配置
        self._validate_config(self.config)
        
        # 应用环境变量覆盖
        self._apply_env_overrides(self.config)
        
        return self.config
    
    def _load_from_file(self, config_file: str) -> Config:
        """从文件加载配置"""
        try:
            file_path = Path(config_file)
            
            if file_path.suffix.lower() == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                raise ConfigError(f"不支持的配置文件格式: {file_path.suffix}")
            
            # 转换为配置对象
            return self._dict_to_config(data)
            
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {e}")
    
    def _dict_to_config(self, data: Dict[str, Any]) -> Config:
        """将字典转换为配置对象"""
        try:
            model_data = data.get('model', {})
            training_data = data.get('training', {})
            data_data = data.get('data', {})
            system_data = data.get('system', {})
            
            return Config(
                model=ModelConfig(**model_data),
                training=TrainingConfig(**training_data),
                data=DataConfig(**data_data),
                system=SystemConfig(**system_data)
            )
        except Exception as e:
            raise ConfigError(f"配置格式错误: {e}")
    
    def _validate_config(self, config: Config):
        """验证配置"""
        errors = []
        
        # 验证模型配置
        if config.model.d_model <= 0:
            errors.append("model.d_model 必须大于0")
        
        if config.model.num_layers <= 0:
            errors.append("model.num_layers 必须大于0")
        
        if config.model.num_heads <= 0:
            errors.append("model.num_heads 必须大于0")
        
        if config.model.d_model % config.model.num_heads != 0:
            errors.append("model.d_model 必须能被 model.num_heads 整除")
        
        # 验证训练配置
        if config.training.batch_size <= 0:
            errors.append("training.batch_size 必须大于0")
        
        if config.training.learning_rate <= 0:
            errors.append("training.learning_rate 必须大于0")
        
        if config.training.num_epochs <= 0:
            errors.append("training.num_epochs 必须大于0")
        
        # 验证数据配置
        if not os.path.exists(config.data.data_dir):
            errors.append(f"数据目录不存在: {config.data.data_dir}")
        
        # 如果有错误，抛出异常
        if errors:
            raise ConfigError("配置验证失败:\n" + "\n".join(f"- {error}" for error in errors))
    
    def _apply_env_overrides(self, config: Config):
        """应用环境变量覆盖"""
        # 支持的环境变量映射
        env_mappings = {
            'SYRIAC_BATCH_SIZE': ('training', 'batch_size', int),
            'SYRIAC_LEARNING_RATE': ('training', 'learning_rate', float),
            'SYRIAC_NUM_EPOCHS': ('training', 'num_epochs', int),
            'SYRIAC_DATA_DIR': ('data', 'data_dir', str),
            'SYRIAC_DEVICE': ('system', 'device', str),
            'SYRIAC_SAVE_DIR': ('system', 'save_dir', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = type_func(os.environ[env_var])
                    setattr(getattr(config, section), key, value)
                    log_info(f"环境变量覆盖: {env_var} = {value}", "配置管理")
                except ValueError as e:
                    log_warning(f"环境变量 {env_var} 格式错误: {e}", "配置管理")
    
    def save_config(self, config: Config, output_file: str, format: str = "yaml"):
        """
        保存配置到文件
        
        参数：
            config (Config): 配置对象
            output_file (str): 输出文件路径
            format (str): 输出格式 ("yaml" 或 "json")
        """
        try:
            # 转换为字典
            config_dict = {
                'model': asdict(config.model),
                'training': asdict(config.training),
                'data': asdict(config.data),
                'system': asdict(config.system)
            }
            
            # 保存到文件
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            elif format.lower() in ["yaml", "yml"]:
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            else:
                raise ConfigError(f"不支持的输出格式: {format}")
            
            log_info(f"配置已保存到: {output_file}", "配置管理")
            
        except Exception as e:
            raise ConfigError(f"保存配置失败: {e}")
    
    def get_model_config_template(self, model_type: str) -> Config:
        """获取模型配置模板"""
        if model_type in self._default_configs:
            return self._default_configs[model_type]
        else:
            raise ConfigError(f"不支持的模型类型: {model_type}")
    
    def create_config_file(self, model_type: str, output_file: str, format: str = "yaml"):
        """创建配置文件模板"""
        config = self.get_model_config_template(model_type)
        self.save_config(config, output_file, format)

    def validate_config(self, config: Dict) -> List[str]:
        """
        验证配置文件的完整性和正确性
        
        参数:
            config (Dict): 配置字典
            
        返回:
            List[str]: 错误信息列表，空列表表示配置有效
        """
        errors = []
        
        # 检查必需的顶级键
        required_sections = ['model', 'training', 'data', 'system']
        for section in required_sections:
            if section not in config:
                errors.append(f"缺少必需的配置节: {section}")
        
        # 检查model节
        if 'model' in config:
            model_config = config['model']
            required_model_keys = ['model_type']
            for key in required_model_keys:
                if key not in model_config:
                    errors.append(f"model节缺少必需参数: {key}")
            
            # 验证模型类型
            if 'model_type' in model_config:
                valid_models = ['transformer', 'bert', 'bilstm', 'diffusion']
                if model_config['model_type'] not in valid_models:
                    errors.append(f"无效的模型类型: {model_config['model_type']}")
        
        # 检查training节
        if 'training' in config:
            training_config = config['training']
            required_training_keys = ['batch_size', 'learning_rate', 'num_epochs']
            for key in required_training_keys:
                if key not in training_config:
                    errors.append(f"training节缺少必需参数: {key}")
            
            # 验证数值参数
            if 'batch_size' in training_config:
                if not isinstance(training_config['batch_size'], int) or training_config['batch_size'] <= 0:
                    errors.append("batch_size必须是正整数")
            
            if 'learning_rate' in training_config:
                if not isinstance(training_config['learning_rate'], (int, float)) or training_config['learning_rate'] <= 0:
                    errors.append("learning_rate必须是正数")
            
            if 'num_epochs' in training_config:
                if not isinstance(training_config['num_epochs'], int) or training_config['num_epochs'] <= 0:
                    errors.append("num_epochs必须是正整数")
        
        # 检查data节
        if 'data' in config:
            data_config = config['data']
            if 'data_dir' in data_config:
                if not os.path.exists(data_config['data_dir']):
                    errors.append(f"数据目录不存在: {data_config['data_dir']}")
        
        # 检查system节
        if 'system' in config:
            system_config = config['system']
            if 'device' in system_config:
                valid_devices = ['auto', 'cpu', 'cuda', 'mps']
                if system_config['device'] not in valid_devices:
                    errors.append(f"无效的设备类型: {system_config['device']}")
        
        return errors


# 全局配置管理器实例
global_config_manager = ConfigManager()


def load_config(config_file: Optional[str] = None, model_type: str = "transformer") -> Config:
    """全局配置加载函数"""
    return global_config_manager.load_config(config_file, model_type)


def save_config(config: Config, output_file: str, format: str = "yaml"):
    """全局配置保存函数"""
    global_config_manager.save_config(config, output_file, format)


if __name__ == "__main__":
    # 测试配置管理器
    print("=== 🧪 配置管理器测试 ===")
    
    manager = ConfigManager()
    
    # 测试加载默认配置
    for model_type in ["transformer", "bert", "bilstm", "diffusion"]:
        print(f"\n📋 {model_type} 默认配置:")
        config = manager.load_config(model_type=model_type)
        print(f"  模型维度: {config.model.d_model}")
        print(f"  层数: {config.model.num_layers}")
        print(f"  批次大小: {config.training.batch_size}")
        print(f"  学习率: {config.training.learning_rate}")
    
    # 测试保存配置
    config = manager.load_config(model_type="transformer")
    output_file = "./test_config.yaml"
    
    try:
        manager.save_config(config, output_file)
        print(f"\n💾 配置已保存到: {output_file}")
        
        # 测试重新加载
        loaded_config = manager.load_config(output_file)
        print("✅ 配置重新加载成功")
        
        # 清理测试文件
        os.remove(output_file)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("✅ 配置管理器测试完成")