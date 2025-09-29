"""
叙利亚文形态分析的数据处理工具模块
整合数据加载、字符映射、数据增强等功能

这个文件的主要功能：
1. 字符映射：叙利亚文字符与数字索引的转换
2. 数据增强：处理不平衡数据的增强策略
3. 数据加载：SyriacDataset类来加载和处理叙利亚文数据
4. 数据批处理和填充功能
5. 创建PyTorch DataLoader进行训练
6. 数据统计和分析功能

数据格式：
- 输入文件(.in): 每行包含叙利亚文字符序列
- 输出文件(.out): 每行包含对应的形态标签序列(数字)
- 每个字符对应一个形态标签
"""

# 导入必要的库
import torch              # PyTorch深度学习框架
import torch.nn as nn     # PyTorch神经网络模块
from torch.utils.data import Dataset, DataLoader  # PyTorch数据处理工具
import numpy as np        # 数值计算库
import os                 # 操作系统接口
import pickle             # 序列化工具
import hashlib            # 哈希计算
import mmap               # 内存映射文件
from concurrent.futures import ThreadPoolExecutor, as_completed  # 多线程处理
from typing import List, Tuple, Dict, Optional  # 类型注解
import threading          # 线程同步
import random             # 随机数生成
from collections import Counter  # 计数器
from sklearn.utils import resample  # 重采样

# 导入统一核心模块
from models.core import (
    DataLoadError,              # 数据加载错误
    log_info,                   # 信息日志函数
    log_warning,                # 警告日志函数
    error_handler_decorator     # 错误处理装饰器
)


# =============================================================================
# 字符映射 / Character Mapping
# =============================================================================

# 核心映射：字符到索引（基于character.md规范）
CHAR_TO_IDX = {
    ' ': 0,    # 空格字符，索引为0
    '>': 1,    # 大于号，叙利亚文中的特殊字符
    'B': 2,    'G': 3,    'D': 4,    'H': 5,    'W': 6,    'Z': 7,    'X': 8,    'V': 9,    'J': 10,
    'K': 11,   'L': 12,   'M': 13,   'N': 14,   'S': 15,   '<': 16,   'P': 17,   'Y': 18,   'Q': 19,
    'R': 20,   'C': 21,   'T': 22,   '^': 23,   '#': 24,   'A': 25,
    # 数字字符
    '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35,
    # 特殊标记
    '<PAD>': 36,   # 填充标记
    '<UNK>': 37,   # 未知字符标记
    '<START>': 38, # 序列开始标记
    '<END>': 39,   # 序列结束标记
}

# 反向映射：索引到字符
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# 词汇表大小
VOCAB_SIZE = len(CHAR_TO_IDX)

# 为了兼容性，保留旧的变量名
FULL_CHAR_TO_IDX = CHAR_TO_IDX
FULL_IDX_TO_CHAR = IDX_TO_CHAR
FULL_VOCAB_SIZE = VOCAB_SIZE

# =============================================================================
# 模型特定词汇表 / Model-Specific Vocabularies
# =============================================================================

# 传统模型词汇表 (只包含基础字符，不包含数字和特殊标记)
BASIC_CHAR_TO_IDX = {k: v for k, v in CHAR_TO_IDX.items() if v < 26}
BASIC_IDX_TO_CHAR = {v: k for k, v in BASIC_CHAR_TO_IDX.items()}
BASIC_VOCAB_SIZE = len(BASIC_CHAR_TO_IDX)

def get_vocab_for_model(model_type: str):
    """
    根据模型类型返回相应的词汇表
    
    参数:
        model_type (str): 模型类型 ('diffusion' 或其他)
        
    返回:
        tuple: (char_to_idx, idx_to_char, vocab_size)
    """
    if model_type.lower() in ['diffusion', 'mdlm']:
        # Diffusion和MDLM模型使用全部40个字符
        return FULL_CHAR_TO_IDX, FULL_IDX_TO_CHAR, FULL_VOCAB_SIZE
    else:
        # 其他模型（BERT, Transformer, BiLSTM）只使用前26个字符
        return BASIC_CHAR_TO_IDX, BASIC_IDX_TO_CHAR, BASIC_VOCAB_SIZE


def parse_interleaved_labels(interleaved_sequence: str) -> List[int]:
    """
    解析字符插值格式的标签序列
    
    这个函数处理扩散模型输出的字符插值格式，例如：
    输入: "0W1B1H0L0J0N0" 
    输出: [0, 1, 1, 0, 0, 0, 0] (每个字符对应的标签)
    
    参数:
        interleaved_sequence (str): 字符插值序列，标签和字符交替出现
        
    返回:
        List[int]: 解析出的标签列表
    """
    if not interleaved_sequence.strip():
        return []
    
    try:
        labels = []
        current_label = ""
        i = 0
        
        while i < len(interleaved_sequence):
            char = interleaved_sequence[i]
            
            # 如果是数字，累积构建标签
            if char.isdigit():
                current_label += char
            else:
                # 遇到非数字字符，说明当前标签结束
                if current_label:
                    labels.append(int(current_label))
                    current_label = ""
                # 跳过字符本身，继续寻找下一个标签
            
            i += 1
        
        # 处理序列末尾的标签
        if current_label:
            labels.append(int(current_label))
            
        return labels
        
    except ValueError as e:
        log_warning(f"标签解析错误: {e}, 序列: {interleaved_sequence[:50]}...", "数据解析")
        # 返回空列表，让调用方处理
        return []


def parse_space_separated_labels(space_separated: str) -> List[int]:
    """
    解析空格分隔的标签格式
    
    这个函数处理传统的空格分隔标签格式，例如：
    输入: "0 1 1 0 0 0 0"
    输出: [0, 1, 1, 0, 0, 0, 0]
    
    参数:
        space_separated (str): 空格分隔的标签字符串
        
    返回:
        List[int]: 解析出的标签列表
    """
    if not space_separated.strip():
        return []
    
    try:
        return [int(x) for x in space_separated.split() if x.strip()]
    except ValueError as e:
        log_warning(f"标签解析错误: {e}, 序列: {space_separated[:50]}...", "数据解析")
        return []


def smart_parse_labels(label_line: str) -> List[int]:
    """
    智能解析标签序列，自动检测格式类型
    
    支持两种格式：
    1. 字符插值格式: "0W1B1H0L0J0N0" (扩散模型输出)
    2. 空格分隔格式: "0 1 1 0 0 0 0" (传统格式)
    
    参数:
        label_line (str): 标签行字符串
        
    返回:
        List[int]: 解析出的标签列表
    """
    if not label_line.strip():
        return []
    
    # 检查是否包含字母，如果有则是字符插值格式
    if any(c.isalpha() or c in '<>' for c in label_line):
        # 字符插值格式：字符和标签交替
        return parse_interleaved_labels(label_line)
    else:
        # 空格分隔格式：纯数字用空格分隔
        return parse_space_separated_labels(label_line)


def char_to_idx(char: str) -> int:
    """
    将字符转换为索引
    
    参数：
        char (str): 输入字符
        
    返回：
        int: 字符对应的索引，未知字符返回<UNK>的索引
    """
    return CHAR_TO_IDX.get(char, CHAR_TO_IDX['<UNK>'])


def idx_to_char(idx: int) -> str:
    """
    将索引转换为字符
    
    参数：
        idx (int): 输入索引
        
    返回：
        str: 索引对应的字符，未知索引返回<UNK>
    """
    return IDX_TO_CHAR.get(idx, '<UNK>')


def encode_sequence(sequence: str) -> List[int]:
    """
    将字符序列编码为索引序列
    
    参数：
        sequence (str): 输入字符序列
        
    返回：
        List[int]: 索引序列
    """
    return [char_to_idx(char) for char in sequence]


def decode_sequence(indices: List[int]) -> str:
    """
    将索引序列解码为字符序列
    
    参数：
        indices (List[int]): 索引序列
        
    返回：
        str: 字符序列
    """
    return ''.join(idx_to_char(idx) for idx in indices)


def validate_character_coverage(text: str) -> Tuple[float, List[str]]:
    """
    验证字符覆盖率
    
    参数：
        text (str): 输入文本
        
    返回：
        Tuple[float, List[str]]: (覆盖率, 未知字符列表)
    """
    unknown_chars = []
    total_chars = len(text)
    unknown_count = 0
    
    for char in text:
        if char not in CHAR_TO_IDX:
            if char not in unknown_chars:
                unknown_chars.append(char)
            unknown_count += 1
    
    coverage = (total_chars - unknown_count) / total_chars if total_chars > 0 else 1.0
    return coverage, unknown_chars


# =============================================================================
# 数据增强 / Data Augmentation
# =============================================================================

class DataAugmentation:
    """
    数据增强类，用于处理极度不平衡的序列标注数据
    """
    
    def __init__(self, min_samples_per_class: int = 100, max_augmentation_factor: float = 5.0):
        """
        初始化数据增强器
        
        参数：
            min_samples_per_class: 每个类别的最小样本数
            max_augmentation_factor: 最大增强倍数
        """
        self.min_samples_per_class = min_samples_per_class
        self.max_augmentation_factor = max_augmentation_factor
    
    def analyze_class_distribution(self, labels: List[List[int]]) -> Dict[int, int]:
        """
        分析类别分布
        
        参数：
            labels: 标签列表
            
        返回：
            类别分布字典
        """
        label_counts = Counter()
        for label_seq in labels:
            for label in label_seq:
                if label != -100:  # 忽略填充标记
                    label_counts[label] += 1
        
        return dict(label_counts)
    
    def get_minority_classes(self, class_distribution: Dict[int, int]) -> List[int]:
        """
        获取少数类别
        
        参数：
            class_distribution: 类别分布字典
            
        返回：
            少数类别列表
        """
        return [cls for cls, count in class_distribution.items() 
                if count < self.min_samples_per_class]
    
    def augment_sequence(self, sequence: List[int], labels: List[int]) -> Tuple[List[int], List[int]]:
        """
        增强单个序列
        
        参数：
            sequence: 输入序列
            labels: 标签序列
            
        返回：
            增强后的序列和标签
        """
        # 简单的增强策略：随机替换一些字符但保持标签不变
        aug_sequence = sequence.copy()
        aug_labels = labels.copy()
        
        # 随机替换少量字符
        if len(sequence) > 2:
            num_replacements = max(1, len(sequence) // 10)
            positions = random.sample(range(len(sequence)), num_replacements)
            
            for pos in positions:
                # 用相似的字符替换（这里简化为随机字符）
                if sequence[pos] != CHAR_TO_IDX.get('<PAD>', 0):
                    # 在词汇表中随机选择一个字符
                    aug_sequence[pos] = random.randint(1, VOCAB_SIZE - 5)  # 避免特殊标记
        
        return aug_sequence, aug_labels
    
    def oversample_minority_classes(self, sequences: List[List[int]], 
                                  labels: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        对少数类别进行过采样
        
        参数：
            sequences: 输入序列列表
            labels: 标签序列列表
            
        返回：
            增强后的序列和标签列表
        """
        class_distribution = self.analyze_class_distribution(labels)
        minority_classes = self.get_minority_classes(class_distribution)
        
        if not minority_classes:
            log_info("数据增强: 没有发现少数类别，跳过数据增强")
            return sequences, labels
        
        log_info(f"数据增强: 发现 {len(minority_classes)} 个少数类别，开始数据增强")
        
        augmented_sequences = sequences.copy()
        augmented_labels = labels.copy()
        
        for seq, label_seq in zip(sequences, labels):
            # 检查序列是否包含少数类别
            has_minority = any(label in minority_classes for label in label_seq if label != -100)
            
            if has_minority:
                # 计算增强倍数
                minority_count = sum(1 for label in label_seq if label in minority_classes and label != -100)
                augmentation_factor = min(self.max_augmentation_factor, 
                                        max(1, self.min_samples_per_class / max(minority_count, 1)))
                
                # 生成增强样本
                num_augmentations = int(augmentation_factor) - 1
                for _ in range(num_augmentations):
                    aug_seq, aug_labels_seq = self.augment_sequence(seq, label_seq)
                    augmented_sequences.append(aug_seq)
                    augmented_labels.append(aug_labels_seq)
        
        log_info(f"数据增强: 数据增强完成，原始样本: {len(sequences)}, 增强后样本: {len(augmented_sequences)}")
        return augmented_sequences, augmented_labels


def compute_enhanced_class_weights(labels: List[List[int]], num_classes: int) -> torch.Tensor:
    """
    计算增强的类别权重
    
    参数：
        labels: 标签列表
        num_classes: 类别数量
        
    返回：
        类别权重张量
    """
    # 统计类别频率
    class_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for label_seq in labels:
        for label in label_seq:
            if label != -100 and 0 <= label < num_classes:
                class_counts[label] += 1
                total_samples += 1
    
    if total_samples == 0:
        return torch.ones(num_classes)
    
    # 计算平衡权重
    class_weights = total_samples / (num_classes * class_counts.clamp(min=1))
    
    # 限制权重范围，避免极端值
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
    
    return class_weights

class DataCache:
    """
    数据缓存管理器
    
    提供数据缓存、序列化和快速加载功能，避免重复的数据处理。
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        初始化缓存管理器
        
        参数：
            cache_dir (str): 缓存目录路径
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        log_info(f"数据缓存: 缓存目录初始化: {cache_dir}")
    
    def _get_cache_key(self, input_file: str, output_file: str, max_length: int) -> str:
        """
        生成缓存键
        
        参数：
            input_file (str): 输入文件路径
            output_file (str): 输出文件路径
            max_length (int): 最大序列长度
            
        返回：
            str: 缓存键（文件内容和参数的哈希值）
        """
        try:
            # 获取文件修改时间和大小
            input_stat = os.stat(input_file)
            output_stat = os.stat(output_file)
            
            # 使用整数时间戳避免浮点精度问题
            input_mtime = int(input_stat.st_mtime)
            output_mtime = int(output_stat.st_mtime)
            
            # 创建包含文件信息和参数的字符串
            cache_info = f"{input_file}_{input_mtime}_{input_stat.st_size}" \
                        f"_{output_file}_{output_mtime}_{output_stat.st_size}" \
                        f"_maxlen_{max_length}"
            
            # 生成SHA256哈希（比MD5更安全）
            return hashlib.sha256(cache_info.encode('utf-8')).hexdigest()
            
        except (OSError, IOError) as e:
            # 如果文件状态获取失败，使用文件路径和参数的哈希
            log_warning(f"无法获取文件状态，使用简化缓存键: {e}", "缓存键生成")
            fallback_info = f"{input_file}_{output_file}_maxlen_{max_length}"
            return hashlib.sha256(fallback_info.encode('utf-8')).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"dataset_{cache_key}.pkl")
    
    def save_to_cache(self, cache_key: str, data: List[Tuple[List[int], List[int]]]):
        """
        保存数据到缓存
        
        参数：
            cache_key (str): 缓存键
            data: 要缓存的数据
        """
        cache_path = self.get_cache_path(cache_key)
        try:
            # 创建临时文件，确保原子性写入
            temp_path = cache_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 原子性重命名
            os.replace(temp_path, cache_path)
            log_info(f"数据缓存: 数据已缓存到: {cache_path}")
            
        except (IOError, OSError) as e:
            log_warning(f"缓存保存失败（IO错误）: {e}", "数据缓存")
            # 清理临时文件
            temp_path = cache_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except (IOError, OSError):
                    pass
        except Exception as e:
            log_warning(f"缓存保存失败（未知错误）: {e}", "数据缓存")
    
    def load_from_cache(self, cache_key: str) -> Optional[List[Tuple[List[int], List[int]]]]:
        """
        从缓存加载数据
        
        参数：
            cache_key (str): 缓存键
            
        返回：
            缓存的数据，如果不存在则返回None
        """
        cache_path = self.get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            log_info(f"数据缓存: 从缓存加载数据: {cache_path}")
            return data
            
        except (pickle.UnpicklingError, EOFError) as e:
            log_warning(f"缓存文件损坏: {e}", "数据缓存")
            self._remove_corrupted_cache(cache_path)
            return None
        except (IOError, OSError) as e:
            log_warning(f"缓存加载失败（IO错误）: {e}", "数据缓存")
            return None
        except Exception as e:
            log_warning(f"缓存加载失败（未知错误）: {e}", "数据缓存")
            self._remove_corrupted_cache(cache_path)
            return None
    
    def _remove_corrupted_cache(self, cache_path: str):
        """
        安全地删除损坏的缓存文件
        
        参数：
            cache_path (str): 缓存文件路径
        """
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                log_info("数据缓存: " + str(f"已删除损坏的缓存文件: {cache_path}"))
        except (IOError, OSError) as e:
            log_warning(f"无法删除损坏的缓存文件: {e}", "数据缓存")

class SyriacDataset(Dataset):
    """
    叙利亚文数据集类，继承自PyTorch的Dataset类
    用于加载和处理叙利亚文形态分析的训练数据
    
    这个类的主要功能：
    1. 读取输入文件和输出文件
    2. 将字符转换为索引
    3. 将标签转换为数字
    4. 处理长度不匹配的问题
    5. 提供数据访问接口
    """
    
    def __init__(self, input_file: str, output_file: str, max_length: int = 64, 
                 use_cache: bool = False, cache_dir: str = "./cache", model_type: str = 'default',
                 return_line_numbers: bool = False):
        """
        初始化数据集
        
        参数：
            input_file (str): 输入文件路径，包含叙利亚文字符序列
            output_file (str): 输出文件路径，包含形态标签序列
            max_length (int): 序列的最大长度，超过此长度会被截断
            use_cache (bool): 是否使用缓存来加速数据加载
            cache_dir (str): 缓存目录路径
            model_type (str): 模型类型，决定使用哪种词汇表
            return_line_numbers (bool): 是否返回原始行号（用于保持预测顺序）
            
        异常：
            FileNotFoundError: 如果输入或输出文件不存在
            ValueError: 如果文件行数不匹配或数据格式错误
        """
        # 保存文件路径和参数
        self.input_file = input_file     # 输入文件路径
        self.output_file = output_file   # 输出文件路径
        self.return_line_numbers = return_line_numbers  # 是否返回行号
        self.max_length = max_length     # 序列最大长度
        self.use_cache = use_cache       # 是否使用缓存
        self.model_type = model_type     # 模型类型
        
        # 初始化缓存管理器
        self.cache_manager = DataCache(cache_dir) if use_cache else None
        
        # 输入验证
        if not os.path.exists(input_file):
            raise DataLoadError(f"输入文件不存在: {input_file}")
        if not os.path.exists(output_file):
            raise DataLoadError(f"输出文件不存在: {output_file}")
        if max_length <= 0:
            raise DataLoadError(f"最大长度必须大于0，当前值: {max_length}")
        
        # 根据模型类型使用相应的词汇表
        self.char_to_idx, self.idx_to_char, self.vocab_size = get_vocab_for_model(model_type)
        
        # 尝试从缓存加载数据
        cached_data = None
        if self.use_cache and self.cache_manager:
            cache_key = self.cache_manager._get_cache_key(input_file, output_file, max_length)
            cached_data = self.cache_manager.load_from_cache(cache_key)
        
        if cached_data is not None:
            # 从缓存加载数据
            self.processed_data = cached_data
            log_info("数据加载: " + str(f"从缓存加载 {len(cached_data)} 条数据"))
        else:
            # 读取和处理原始数据
            self._load_raw_data()
            self._validate_character_coverage()
            self.processed_data = self._process_data()
            
            # 保存到缓存
            if self.use_cache and self.cache_manager:
                self.cache_manager.save_to_cache(cache_key, self.processed_data)
    
    def _load_raw_data(self):
        """
        高效读取原始数据文件
        
        使用内存映射和优化的文件读取策略
        """
        # 读取输入文件和输出文件
        # 使用utf-8编码确保正确读取叙利亚文字符
        try:
            # 使用内存映射读取大文件（如果文件足够大）
            input_stat = os.stat(self.input_file)
            output_stat = os.stat(self.output_file)
            
            # 如果文件大于1MB，使用内存映射
            if input_stat.st_size > 1024 * 1024:
                self.input_lines = self._read_file_mmap(self.input_file)
            else:
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    # 读取所有行，只去除行尾的换行符，保留行首的空格
                    # 空格是叙利亚文中重要的字符，不应该被去除
                    self.input_lines = [line.rstrip('\n\r') for line in f.readlines() if line.strip()]
            
            if output_stat.st_size > 1024 * 1024:
                self.output_lines = self._read_file_mmap(self.output_file, strip_content=True)
            else:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    # 读取所有行并去除首尾空白字符，过滤空行
                    self.output_lines = [line.strip() for line in f.readlines() if line.strip()]
                    
        except UnicodeDecodeError as e:
            raise DataLoadError(f"文件编码错误，请确保文件使用UTF-8编码: {e}")
        except Exception as e:
            raise DataLoadError(f"读取文件时发生错误: {e}")
        
        # 确保输入文件和输出文件的行数相同
        if len(self.input_lines) != len(self.output_lines):
            raise DataLoadError(
                f"输入文件和输出文件行数不匹配: "
                f"输入文件{len(self.input_lines)}行，输出文件{len(self.output_lines)}行"
            )
        
        if len(self.input_lines) == 0:
            raise DataLoadError("数据文件为空或没有有效数据行")
    
    def _read_file_mmap(self, file_path: str, strip_content: bool = False) -> List[str]:
        """
        使用内存映射读取文件
        
        参数：
            file_path (str): 文件路径
            strip_content (bool): 是否去除行首尾空白字符
            
        返回：
            List[str]: 文件行列表
        """
        lines = []
        
        # 首先尝试内存映射读取
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    log_warning(f"文件为空: {file_path}", "文件读取")
                    return lines
                
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    for line in iter(mmapped_file.readline, b""):
                        try:
                            line_str = line.decode('utf-8')
                            if strip_content:
                                line_str = line_str.strip()
                            else:
                                line_str = line_str.rstrip('\n\r')
                            
                            if line_str:  # 过滤空行
                                lines.append(line_str)
                        except UnicodeDecodeError as e:
                            log_warning(f"行解码失败，跳过: {e}", "文件读取")
                            continue
            
            log_info("文件读取: " + str(f"内存映射读取成功，共{len(lines)}行"))
            return lines
            
        except (OSError, IOError, ValueError) as e:
            log_warning(f"内存映射读取失败，回退到普通读取: {e}", "文件读取")
            
        # 回退到普通读取
        return self._read_file_regular(file_path, strip_content)
    
    def _read_file_regular(self, file_path: str, strip_content: bool = False) -> List[str]:
        """
        常规文件读取
        
        参数：
            file_path (str): 文件路径
            strip_content (bool): 是否去除行首尾空白字符
            
        返回：
            List[str]: 文件行列表
        """
        lines = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if strip_content:
                            line_str = line.strip()
                        else:
                            line_str = line.rstrip('\n\r')
                        
                        if line_str:  # 过滤空行
                            lines.append(line_str)
                    except UnicodeDecodeError as e:
                        log_warning(f"第{line_num}行解码失败，跳过: {e}", "文件读取")
                        continue
            
            log_info("文件读取: " + str(f"常规读取成功，共{len(lines)}行"))
            
        except (IOError, OSError) as e:
            raise DataLoadError(f"无法读取文件 {file_path}: {e}")
        except UnicodeDecodeError as e:
            raise DataLoadError(f"文件编码错误 {file_path}: {e}")
        
        return lines
        
    def _validate_character_coverage(self):
        """
        验证输入数据中的所有字符都在定义的字符集中
        
        这个方法检查输入文件中是否有未知字符，如果有则发出警告
        """
        # 收集所有未知字符
        all_unknown = set()
        
        # 遍历所有输入行
        for line in self.input_lines:
            # 检查这一行是否有未知字符
            coverage, unknown_chars = validate_character_coverage(line)
            # 将未知字符添加到总集合中
            all_unknown.update(unknown_chars)
        
        # 如果发现未知字符，发出警告
        if all_unknown:
            log_warning(f"发现未知字符: {all_unknown}", "字符验证")
            log_info("字符验证: " + str("这些字符将被映射为<UNK>标记"))
    
    def _process_single_line(self, line_data: Tuple[int, Tuple[str, str]]) -> Optional[Tuple[List[int], List[int]]]:
        """
        处理单行数据（用于并行处理）
        
        参数：
            line_data: (行索引, (输入行, 输出行))
            
        返回：
            处理后的数据或None（如果处理失败）
        """
        line_idx, (input_line, output_line) = line_data
        
        try:
            # 根据模型类型使用不同的字符编码策略
            if self.model_type in ['bert', 'bilstm', 'transformer']:
                # 基础模型只允许基础词表字符，特殊标记映射为空格(0)
                char_indices = []
                for char in input_line:
                    if char in self.char_to_idx:
                        char_indices.append(self.char_to_idx[char])
                    else:
                        # 对于基础模型，未知字符（包括特殊标记）映射为空格
                        char_indices.append(0)
            else:
                # diffusion等模型允许全部字符，使用UNK处理未知字符
                char_indices = [self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 0)) for char in input_line]
            
            # 智能解析标签序列，支持字符插值格式和空格分隔格式
            try:
                labels = smart_parse_labels(output_line)
                if not labels:  # 如果解析失败返回空列表
                    log_warning(f"第{line_idx + 1}行标签解析失败，跳过该行", "数据处理")
                    return None
            except Exception as e:
                log_warning(f"第{line_idx + 1}行标签格式错误: {e}", "数据处理")
                return None
            
            # 确保输入和输出长度匹配
            if len(char_indices) != len(labels):
                warning_msg = f"第{line_idx + 1}行长度不匹配: 输入{len(char_indices)}个字符, 输出{len(labels)}个标签"
                log_warning(warning_msg, "数据对齐")
                
                # 取较短的长度，确保字符和标签一一对应
                min_length = min(len(char_indices), len(labels))
                if min_length == 0:  # 如果为空，跳过这行
                    return None
                char_indices = char_indices[:min_length]
                labels = labels[:min_length]
            
            # 如果序列过长，进行截断
            if len(char_indices) > self.max_length:
                char_indices = char_indices[:self.max_length]
                labels = labels[:self.max_length]
            
            # 增强的数据验证
            # 1. 验证标签范围（假设标签应该是非负数）
            if any(label < 0 for label in labels):
                log_warning(f"第{line_idx + 1}行包含负数标签，将跳过", "数据验证")
                return None
            
            # 2. 验证标签的最大值是否合理（假设标签不应该超过10000）
            max_label = max(labels) if labels else 0
            if max_label > 10000:
                log_warning(f"第{line_idx + 1}行包含异常大的标签值 {max_label}，将跳过", "数据验证")
                return None
            
            # 3. 验证序列长度是否合理
            if len(char_indices) > 5000:  # 异常长的序列
                log_warning(f"第{line_idx + 1}行序列过长 ({len(char_indices)} 字符)，将跳过", "数据验证")
                return None
            
            # 4. 验证字符索引是否在有效范围内
            if any(idx < 0 or idx >= self.vocab_size for idx in char_indices):
                log_warning(f"第{line_idx + 1}行包含无效的字符索引，将跳过", "数据验证")
                return None
            
            # 如果需要返回行号，则包含行号
            if self.return_line_numbers:
                return (char_indices, labels, line_idx)
            else:
                return (char_indices, labels)
            
        except Exception as e:
            log_warning(f"处理第{line_idx + 1}行时发生错误: {e}", "数据处理")
            return None

    def _process_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        处理原始数据，将字符和标签转换为索引列表
        
        使用线程安全的并行处理来提高性能
        
        返回：
            List[Tuple[List[int], List[int]]]: 处理后的数据列表
            每个元素是(字符索引列表, 标签列表)的元组
            
        异常：
            ValueError: 如果数据格式错误或标签无法转换为整数
        """
        log_info("数据处理: " + str("开始处理数据..."))
        
        # 准备数据用于并行处理
        line_data = [(i, (input_line, output_line)) 
                    for i, (input_line, output_line) 
                    in enumerate(zip(self.input_lines, self.output_lines))]
        
        processed = []  # 存储处理后的数据
        errors_count = 0  # 错误计数
        
        # 决定是否使用并行处理
        total_lines = len(line_data)
        use_parallel = total_lines > 1000  # 超过1000行时使用并行处理
        
        if use_parallel:
            # 使用线程池进行并行处理
            max_workers = min(4, os.cpu_count() or 1)  # 最多使用4个线程
            log_info("数据处理: " + str(f"使用{max_workers}个线程并行处理{total_lines}行数据"))
            
            # 使用线程安全的结果收集
            processed_lock = threading.Lock()
            errors_lock = threading.Lock()
            
            def process_and_collect(line_data_chunk):
                """处理数据块并收集结果"""
                local_processed = []
                local_errors = 0
                
                for line in line_data_chunk:
                    try:
                        result = self._process_single_line(line)
                        if result is not None:
                            local_processed.append(result)
                        else:
                            local_errors += 1
                    except Exception as e:
                        local_errors += 1
                        log_warning(f"处理第{line[0]+1}行时发生错误: {e}", "数据处理")
                
                return local_processed, local_errors
            
            # 将数据分块以提高并行效率
            chunk_size = max(1, total_lines // (max_workers * 4))
            chunks = [line_data[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_chunk = {executor.submit(process_and_collect, chunk): chunk 
                                  for chunk in chunks}
                
                # 收集结果
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_processed, chunk_errors = future.result()
                        processed.extend(chunk_processed)
                        errors_count += chunk_errors
                    except Exception as e:
                        errors_count += len(future_to_chunk[future])
                        log_warning(f"处理数据块时发生错误: {e}", "数据处理")
        else:
            # 顺序处理
            log_info("数据处理: " + str(f"顺序处理{total_lines}行数据"))
            for line in line_data:
                try:
                    result = self._process_single_line(line)
                    if result is not None:
                        processed.append(result)
                    else:
                        errors_count += 1
                except Exception as e:
                    errors_count += 1
                    log_warning(f"处理第{line[0]+1}行时发生错误: {e}", "数据处理")
        
        if len(processed) == 0:
            raise DataLoadError("没有成功处理任何数据行，请检查数据格式")
        
        # 打印处理统计信息
        success_rate = len(processed) / total_lines * 100
        log_info("数据统计: " + str(f"数据处理完成: {len(processed)}/{total_lines} 行成功处理 ({success_rate:.1f}%)"))
        if errors_count > 0:
            log_warning(f"跳过了 {errors_count} 行错误数据", "数据统计")
            
        return processed
    
    def __len__(self) -> int:
        """
        返回数据集的大小
        
        返回：
            int: 数据集中样本的数量
        """
        return len(self.processed_data)
    
    def __getitem__(self, idx: int):
        """
        获取数据集中的单个样本
        
        参数：
            idx (int): 样本索引
            
        返回：
            如果return_line_numbers=False:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (输入张量, 标签张量, 注意力掩码张量)
            如果return_line_numbers=True:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: 
                (输入张量, 标签张量, 注意力掩码张量, 行号)
        """
        # 获取指定索引的数据
        if self.return_line_numbers:
            char_indices, labels, line_num = self.processed_data[idx]
        else:
            char_indices, labels = self.processed_data[idx]
        
        # 转换为PyTorch张量
        # dtype=torch.long表示长整型，适合索引数据
        input_tensor = torch.tensor(char_indices, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 创建注意力掩码
        # 对于真实的令牌，掩码值为1；对于填充令牌，掩码值为0
        attention_mask = torch.ones(len(char_indices), dtype=torch.long)
        
        if self.return_line_numbers:
            return input_tensor, label_tensor, attention_mask, line_num
        else:
            return input_tensor, label_tensor, attention_mask

def collate_fn_with_line_numbers(batch):
    """
    处理带行号的批次数据
    
    参数：
        batch: 包含(input, labels, attention_mask, line_num)的列表
        
    返回：
        (padded_input, padded_labels, attention_mask, line_numbers)
    """
    # 分离数据和行号
    inputs = []
    labels = []
    masks = []
    line_numbers = []
    
    for item in batch:
        if len(item) == 4:  # 包含行号
            inputs.append(item[0])
            labels.append(item[1])
            masks.append(item[2])
            line_numbers.append(item[3])
        else:  # 不包含行号
            inputs.append(item[0])
            labels.append(item[1])
            masks.append(item[2])
    
    # 使用原始的collate_fn处理数据
    padded_batch = collate_fn([(inp, lab, mask) for inp, lab, mask in zip(inputs, labels, masks)])
    
    if line_numbers:
        # 返回带行号的批次
        return padded_batch[0], padded_batch[1], padded_batch[2], torch.tensor(line_numbers, dtype=torch.long)
    else:
        return padded_batch

def collate_fn(batch):
    """
    高效的批处理函数，用于将多个样本组合成一个批次
    
    这个函数的主要作用：
    1. 将不同长度的序列填充到相同长度
    2. 创建批次张量
    3. 正确处理填充标记
    4. 使用优化的张量操作提高性能
    
    参数：
        batch: 一个批次的样本列表，每个样本是(输入, 标签, 掩码)的元组
        
    返回：
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        (填充后的输入批次, 填充后的标签批次, 填充后的掩码批次)
    """
    # 将批次中的输入、标签和掩码分离
    inputs, labels, attention_masks = zip(*batch)
    
    # 找到批次中最长序列的长度
    max_len = max(len(inp) for inp in inputs)
    batch_size = len(batch)
    
    # 根据数据中的最大索引值确定pad值
    # 如果数据中最大索引小于26，说明是基础模型，使用0作为pad值
    # 否则使用<PAD>标记（36）
    max_index_in_batch = max(torch.max(inp).item() for inp in inputs)
    if max_index_in_batch < 26:
        pad_token_id = 0  # 基础模型使用空格作为pad
    else:
        pad_token_id = FULL_CHAR_TO_IDX['<PAD>']  # diffusion等模型使用<PAD>
    
    # 创建预填充的张量
    padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    # 高效填充每个样本
    for i, (inp, lab, mask) in enumerate(zip(inputs, labels, attention_masks)):
        seq_len = len(inp)
        padded_inputs[i, :seq_len] = inp
        padded_labels[i, :seq_len] = lab
        padded_masks[i, :seq_len] = mask
    
    return padded_inputs, padded_labels, padded_masks


class SmartBatchSampler:
    """
    智能批处理采样器
    
    根据序列长度对数据进行分组，减少填充开销
    """
    
    def __init__(self, dataset, batch_size: int, max_length: int, drop_last: bool = False):
        """
        初始化智能批处理采样器
        
        参数：
            dataset: 数据集
            batch_size: 批次大小
            max_length: 最大序列长度
            drop_last: 是否丢弃最后一个不完整的批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_last = drop_last
        
        # 按序列长度分组
        self.length_groups = self._group_by_length()
        
    def _group_by_length(self) -> Dict[int, List[int]]:
        """
        按序列长度对数据进行分组
        
        返回：
            Dict[int, List[int]]: 长度到索引列表的映射
        """
        length_groups = {}
        
        for idx in range(len(self.dataset)):
            seq_len = len(self.dataset.processed_data[idx][0])  # 获取序列长度
            
            # 按长度区间分组（每32个长度为一组）
            length_bucket = (seq_len // 32) * 32
            
            if length_bucket not in length_groups:
                length_groups[length_bucket] = []
            
            length_groups[length_bucket].append(idx)
        
        return length_groups
    
    def __iter__(self):
        """
        迭代生成批次
        """
        # 随机打乱每个长度组内的顺序
        import random
        
        all_batches = []
        
        for length_bucket, indices in self.length_groups.items():
            # 打乱组内顺序
            random.shuffle(indices)
            
            # 分割成批次
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                # 检查是否丢弃最后一个不完整的批次
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                all_batches.append(batch)
        
        # 随机打乱批次顺序
        random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """
        返回批次数量
        """
        total_batches = 0
        for indices in self.length_groups.values():
            num_batches = len(indices) // self.batch_size
            if len(indices) % self.batch_size != 0 and not self.drop_last:
                num_batches += 1
            total_batches += num_batches
        
        return total_batches

def create_data_loaders(train_input: str, train_output: str, 
                       val_input: str, val_output: str,
                       test_input: str, test_output: str,
                       batch_size: int = 32, max_length: int = 64,
                       use_cache: bool = False, cache_dir: str = "./cache",
                       use_smart_batching: bool = True, num_workers: int = 0,
                       model_type: str = 'default',
                       test_return_line_numbers: bool = False):
    """
    创建训练、验证和测试的数据加载器（优化版）
    
    参数：
        train_input (str): 训练集输入文件路径
        train_output (str): 训练集输出文件路径
        val_input (str): 验证集输入文件路径
        val_output (str): 验证集输出文件路径
        test_input (str): 测试集输入文件路径
        test_output (str): 测试集输出文件路径
        batch_size (int): 批次大小
        max_length (int): 序列最大长度
        use_cache (bool): 是否使用数据缓存
        cache_dir (str): 缓存目录
        use_smart_batching (bool): 是否使用智能批处理（按长度分组）
        num_workers (int): 数据加载器的工作线程数
        model_type (str): 模型类型，决定使用哪种词汇表
        test_return_line_numbers (bool): 测试集是否返回行号（用于保持预测顺序）
        
    返回：
        Tuple: (训练数据加载器, 验证数据加载器, 测试数据加载器, 字符映射字典)
    """
    log_info("数据加载: 创建优化的数据加载器...")
    
    # 根据模型类型获取正确的词汇表
    char_to_idx, idx_to_char, vocab_size = get_vocab_for_model(model_type)
    log_info(f"数据加载: 使用 {model_type} 模型的词汇表，大小: {vocab_size}")
    
    # 创建数据集，使用指定模型的词汇表
    train_dataset = SyriacDataset(train_input, train_output, max_length=max_length, 
                                 use_cache=use_cache, cache_dir=cache_dir, model_type=model_type)
    val_dataset = SyriacDataset(val_input, val_output, max_length=max_length, 
                               use_cache=use_cache, cache_dir=cache_dir, model_type=model_type)
    # 测试数据集可能需要返回行号
    test_dataset = SyriacDataset(test_input, test_output, max_length=max_length, 
                                use_cache=use_cache, cache_dir=cache_dir, model_type=model_type,
                                return_line_numbers=test_return_line_numbers)
    
    # 设置数据加载器参数
    # 测试集如果需要返回行号，使用特殊的collate函数
    test_collate_fn = collate_fn_with_line_numbers if test_return_line_numbers else collate_fn
    
    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),  # 如果使用GPU，启用pin_memory
        'persistent_workers': num_workers > 0,   # 如果使用多进程，启用persistent_workers
    }
    
    # 创建数据加载器
    if use_smart_batching:
        # 使用智能批处理采样器
        log_info("数据加载: " + str("使用智能批处理采样器"))
        
        # 训练集使用智能批处理
        train_sampler = SmartBatchSampler(train_dataset, batch_size, max_length, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, **dataloader_kwargs)
        
        # 验证集和测试集使用常规批处理
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **dataloader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn, **dataloader_kwargs)
    else:
        # 使用常规批处理
        log_info("数据加载: " + str("使用常规批处理"))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, **dataloader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **dataloader_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn, **dataloader_kwargs)
    
    # 打印性能优化信息
    log_info("数据加载: " + str(f"数据加载器优化设置:"))
    log_info("数据加载: " + str(f"  缓存: {'启用' if use_cache else '禁用'}"))
    log_info("数据加载: " + str(f"  智能批处理: {'启用' if use_smart_batching else '禁用'}"))
    log_info("数据加载: " + str(f"  工作线程数: {num_workers}"))
    log_info("数据加载: " + str(f"  内存锁定: {'启用' if torch.cuda.is_available() else '禁用'}"))
    
    return train_loader, val_loader, test_loader, char_to_idx

def get_num_classes(output_files: List[str] = None) -> int:
    """
    优先从实际数据文件中获取类别数量，确保准确性
    
    参数：
        output_files (List[str]): 输出文件路径列表
        
    返回：
        int: 类别数量（实际数据中的最大标签值+1）
    """
    # 优先方法：只从训练文件统计类别数（确保一致性）
    if output_files:
        # 找到训练文件
        train_file = None
        for file_path in output_files:
            if 'train.out' in file_path:
                train_file = file_path
                break
        
        if train_file and os.path.exists(train_file):
            all_labels = set()
            with open(train_file, 'r') as f:
                for line in f:
                    labels = smart_parse_labels(line.strip())
                    all_labels.update(labels)
            
            if all_labels:
                num_classes = max(all_labels) + 1
                log_info("数据统计: " + str(f"✅ 从训练数据统计到 {num_classes} 个类别（标签范围: {min(all_labels)} ~ {max(all_labels)}）"))
                log_info("数据统计: " + str("📝 注意：使用训练集类别数，验证时可能遇到训练集独有类别"))
                return num_classes
    
    # 备用方法：从data/patterns.csv读取
    patterns_file = "data/patterns.csv"
    if os.path.exists(patterns_file):
        try:
            import pandas as pd
            df = pd.read_csv(patterns_file)
            max_label = max(0, df['label'].max())
            num_classes = max_label + 1
            log_warning(f"⚠️  从patterns.csv读取到 {num_classes} 个类别，但建议使用实际数据统计", "数据统计")
            return num_classes
        except Exception as e:
            log_warning(f"读取patterns.csv失败: {e}", "数据统计")
    
    # 默认值
    log_warning("❌ 无法确定类别数量，使用默认值309", "数据统计")
    return 309


def get_class_weights(output_files: List[str], num_classes: int) -> torch.Tensor:
    """
    从输出文件中计算类别权重，用于处理数据不平衡问题

    参数：
        output_files (List[str]): 输出文件路径列表
        num_classes (int): 唯一类别的总数

    返回：
        torch.Tensor: 每个类别的权重张量
    """
    all_labels = []
    log_info("数据统计: " + str("⚖️  开始计算类别权重..."))
    for file_path in output_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    labels = smart_parse_labels(line.strip())
                    all_labels.extend(labels)
                except Exception:
                    continue
    
    if not all_labels:
        log_warning("🤷 没有找到任何标签，无法计算权重。", "数据统计")
        return torch.ones(num_classes)

    counts = torch.bincount(torch.tensor(all_labels, dtype=torch.long), minlength=num_classes).float()
    
    # 新的、更稳定的权重计算方案
    # 为频率添加一个平滑项，避免除以零
    counts += 1
    # 权重与频率成反比
    weights = 1.0 / counts
    # 将权重归一化，使最小的权重为1
    weights = weights / weights.min()

    log_info("数据统计: " + str(f"✅ 类别权重计算完成 (新方案)。示例 (前10个): {[f'{w:.2f}' for w in weights[:10]]}"))
    
    return weights


def analyze_data_statistics(data_dir: str):
    """
    分析数据集的统计信息
    
    参数：
        data_dir (str): 数据目录路径
        
    这个函数会分析：
    1. 每个数据集的基本信息（行数、长度分布等）
    2. 字符与标签的对应关系
    3. 字符覆盖率
    4. 数据质量问题
    """
    print("=== 📊 数据统计分析 ===")
    
    # 分析训练集、验证集和测试集
    for split in ['train', 'val', 'test']:
        input_file = os.path.join(data_dir, f'{split}.in')
        output_file = os.path.join(data_dir, f'{split}.out')
        
        # 如果文件不存在，跳过
        if not os.path.exists(input_file) or not os.path.exists(output_file):
            continue
            
        print(f"\n📁 {split.upper()} 数据集:")
        
        # 读取文件
        with open(input_file, 'r', encoding='utf-8') as f:
            input_lines = [line.rstrip('\n\r') for line in f.readlines()]
        
        with open(output_file, 'r', encoding='utf-8') as f:
            output_lines = [line.strip() for line in f.readlines()]
        
        # 基本统计信息
        print(f"  📄 行数: {len(input_lines)}")
        
        # 字符长度统计
        char_lengths = [len(line) for line in input_lines]
        print(f"  📏 字符长度: 平均={np.mean(char_lengths):.1f}, "
              f"最小={min(char_lengths)}, 最大={max(char_lengths)}")
        
        # 标签数量统计
        label_counts = []
        for line in output_lines:
            labels = smart_parse_labels(line.strip())
            label_counts.append(len(labels))
        
        print(f"  🏷️  标签数量: 平均={np.mean(label_counts):.1f}, "
              f"最小={min(label_counts)}, 最大={max(label_counts)}")
        
        # 验证字符与标签对应关系
        mismatches = 0
        for i, (input_line, output_line) in enumerate(zip(input_lines, output_lines)):
            char_count = len(input_line)
            labels = smart_parse_labels(output_line.strip())
            label_count = len(labels)
            if char_count != label_count:
                mismatches += 1
                # 只显示前3个不匹配的例子
                if mismatches <= 3:
                    print(f"  ⚠️  不匹配例子 {i+1}: {char_count}字符 vs {label_count}标签")
        
        if mismatches > 0:
            print(f"  ❌ 字符-标签不匹配: {mismatches}行 ({mismatches/len(input_lines)*100:.1f}%)")
        else:
            print(f"  ✅ 字符-标签完全匹配")
        
        # 字符覆盖率分析
        all_chars = set()
        for line in input_lines:
            all_chars.update(line)
        
        unknown_chars = set()
        for char in all_chars:
            if char not in FULL_CHAR_TO_IDX:
                unknown_chars.add(char)
        
        coverage_rate = (len(all_chars - unknown_chars) / len(all_chars)) * 100
        print(f"  📊 字符覆盖率: {len(all_chars - unknown_chars)}/{len(all_chars)} = {coverage_rate:.1f}%")
        
        if unknown_chars:
            print(f"  ❓ 未知字符: {unknown_chars}")


def performance_test(data_dir: str, batch_size: int = 16):
    """
    数据加载性能测试
    
    参数：
        data_dir (str): 数据目录路径
        batch_size (int): 批次大小
    """
    print("\n=== ⚡ 数据加载性能测试 ===")
    
    import time
    
    # 测试文件路径
    train_in = os.path.join(data_dir, "train.in")
    train_out = os.path.join(data_dir, "train.out")
    val_in = os.path.join(data_dir, "val.in")
    val_out = os.path.join(data_dir, "val.out")
    test_in = os.path.join(data_dir, "test.in")
    test_out = os.path.join(data_dir, "test.out")
    
    # 检查文件是否存在
    files = [train_in, train_out, val_in, val_out, test_in, test_out]
    for file in files:
        if not os.path.exists(file):
            print(f"❌ 文件不存在: {file}")
            return
    
    # 测试配置
    test_configs = [
        {"name": "基础配置", "use_cache": False, "use_smart_batching": False, "num_workers": 0},
        {"name": "启用缓存", "use_cache": True, "use_smart_batching": False, "num_workers": 0},
        {"name": "启用智能批处理", "use_cache": False, "use_smart_batching": True, "num_workers": 0},
        {"name": "多线程处理", "use_cache": False, "use_smart_batching": False, "num_workers": 2},
        {"name": "全部优化", "use_cache": True, "use_smart_batching": True, "num_workers": 2},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 测试配置: {config['name']}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 创建数据加载器
            train_loader, val_loader, test_loader, _ = create_data_loaders(
                train_in, train_out, val_in, val_out, test_in, test_out,
                batch_size=batch_size, **{k: v for k, v in config.items() if k != 'name'}
            )
            
            # 记录创建时间
            create_time = time.time() - start_time
            
            # 测试训练数据加载速度
            batch_count = 0
            batch_start_time = time.time()
            
            for batch_idx, (inputs, labels, masks) in enumerate(train_loader):
                batch_count += 1
                if batch_count >= 10:  # 只测试前10个批次
                    break
            
            batch_time = time.time() - batch_start_time
            
            # 记录结果
            result = {
                'config': config['name'],
                'create_time': create_time,
                'batch_time': batch_time,
                'batches_per_second': batch_count / batch_time if batch_time > 0 else 0,
                'total_batches': len(train_loader)
            }
            
            results.append(result)
            
            print(f"   创建时间: {create_time:.3f}秒")
            print(f"   批次处理时间: {batch_time:.3f}秒 ({batch_count}个批次)")
            print(f"   批次处理速度: {result['batches_per_second']:.2f} 批次/秒")
            print(f"   总批次数: {result['total_batches']}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            continue
    
    # 显示性能对比
    if len(results) > 1:
        print(f"\n📊 性能对比（相对于基础配置）:")
        baseline = results[0]  # 基础配置作为基准
        
        for result in results[1:]:
            create_speedup = baseline['create_time'] / result['create_time'] if result['create_time'] > 0 else 0
            batch_speedup = result['batches_per_second'] / baseline['batches_per_second'] if baseline['batches_per_second'] > 0 else 0
            
            print(f"   {result['config']}:")
            print(f"     创建加速: {create_speedup:.2f}x")
            print(f"     批次处理加速: {batch_speedup:.2f}x")
    
    print(f"\n💡 性能优化建议:")
    print(f"   1. 启用缓存可以显著减少重复数据处理时间")
    print(f"   2. 智能批处理可以减少填充开销，提高GPU利用率")
    print(f"   3. 多线程处理可以在CPU密集型任务中提高并行度")
    print(f"   4. 组合使用多种优化技术可以获得最佳性能")


# 主程序入口
if __name__ == "__main__":
    # 测试数据加载功能
    print("=== 🧪 数据加载测试 ===")
    
    # 数据目录路径（使用相对路径）
    data_dir = "./data"
    
    # 如果相对路径不存在，尝试查找可能的数据目录
    if not os.path.exists(data_dir):
        possible_dirs = [
            "./data",
            "../data", 
            "../../data",
            os.path.join(os.path.dirname(__file__), "data"),
            os.path.join(os.path.dirname(__file__), "..", "data")
        ]
        
        for possible_dir in possible_dirs:
            if os.path.exists(possible_dir):
                data_dir = possible_dir
                break
        else:
            print("❌ 找不到数据目录，请确保数据文件存在于以下位置之一：")
            for dir_path in possible_dirs:
                print(f"   - {os.path.abspath(dir_path)}")
            exit(1)
    
    # 首先分析数据统计
    analyze_data_statistics(data_dir)
    
    # 测试数据加载器
    try:
        print("\n=== 🚀 数据加载器测试 ===")
        
        # 创建数据加载器（使用优化功能）
        train_loader, val_loader, test_loader, char_to_idx = create_data_loaders(
            os.path.join(data_dir, "train.in"),
            os.path.join(data_dir, "train.out"),
            os.path.join(data_dir, "val.in"),
            os.path.join(data_dir, "val.out"),
            os.path.join(data_dir, "test.in"),
            os.path.join(data_dir, "test.out"),
            batch_size=8,  # 小批次便于测试
            use_cache=False,  # 禁用缓存
            use_smart_batching=True,  # 启用智能批处理
            num_workers=2  # 使用2个工作线程
        )
        
        # 显示基本信息
        print(f"📊 词汇表大小: {len(char_to_idx)}")
        print(f"🏋️  训练批次: {len(train_loader)}")
        print(f"🔍 验证批次: {len(val_loader)}")
        print(f"🧪 测试批次: {len(test_loader)}")
        
        # 测试一个批次
        print("\n--- 批次数据示例 ---")
        for batch in train_loader:
            inputs, labels, masks = batch
            print(f"📥 输入形状: {inputs.shape}")
            print(f"🏷️  标签形状: {labels.shape}")
            print(f"👁️  掩码形状: {masks.shape}")
            
            # 解码第一个样本作为示例
            print(f"\n📝 第一个样本:")
            print(f"   输入: '{decode_sequence(inputs[0])}'")
            print(f"   标签: {labels[0]}")
            print(f"   掩码: {masks[0]}")
            
            # 只显示第一个批次
            break
            
        print("\n✅ 数据加载测试完成！")
        
        # 运行性能测试
        performance_test(data_dir, batch_size=8)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请确保数据文件存在于指定目录中")
        import traceback
        traceback.print_exc()