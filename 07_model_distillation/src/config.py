# 导入 dataclass，用来集中管理配置参数
from dataclasses import dataclass
# 导入 Path，用来安全处理 Windows / Linux 路径
from pathlib import Path
# 导入 PyTorch
import torch

# 获取项目根目录：distillation_project
BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


@dataclass
class Config:
    # Teacher 模型目录
    teacher_path: Path = (
        BASE_DIR
        / "models"
        / "teacher"
    )

    # Student 最佳模型保存位置
    student_path: Path = (
        BASE_DIR
        / "models"
        / "student"
        / "best_student.pt"
    )

    # 训练集路径
    train_path: Path = (
        BASE_DIR
        / "data"
        / "train.txt"
    )

    # 验证集路径
    dev_path: Path = (
        BASE_DIR
        / "data"
        / "dev.txt"
    )

    # 测试集路径
    test_path: Path = (
        BASE_DIR
        / "data"
        / "test.txt"
    )

    # 类别名称文件
    class_path: Path = (
        BASE_DIR
        / "data"
        / "class.txt"
    )

    # Teacher Logits 缓存文件
    teacher_logits_cache: Path = (
        BASE_DIR
        / "cache"
        / "train_teacher_logits.pt"
    )

    # Teacher Hidden 缓存文件
    teacher_hidden_cache: Path = (
        BASE_DIR
        / "cache"
        / "train_teacher_hidden.pt"
    )

    # Teacher 缓存对应的原始样本下标
    teacher_indices_cache: Path = (
        BASE_DIR
        / "cache"
        / "train_teacher_indices.pt"
    )

    # 类别数量
    num_classes: int = 10

    # 文本最大 Token 长度
    max_length: int = 128

    # Batch 大小
    batch_size: int = 64

    # 训练轮数
    epochs: int = 10

    # Student 学习率
    learning_rate: float = 1e-3

    # 蒸馏温度
    temperature: float = 2.0

    # 真实标签 CE Loss 权重
    hard_weight: float = 0.4

    # Teacher Soft KL Loss 权重
    soft_weight: float = 0.5

    # Teacher Hidden Feature Loss 权重
    feature_weight: float = 0.1

    # Early Stopping 容忍轮数
    patience: int = 3

    # Student Embedding 维度
    embed_size: int = 128

    # BiLSTM 单方向隐藏维度
    hidden_size: int = 256

    # BiLSTM 层数
    num_layers: int = 2

    # Dropout 比例
    dropout: float = 0.3

    # DataLoader 子进程数量
    num_workers: int = 0

    # 是否优先使用 Teacher 离线缓存
    use_teacher_cache: bool = True


# 创建全局配置对象
config = Config()

# 自动创建 Student 模型目录
config.student_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

# 自动创建缓存目录
config.teacher_logits_cache.parent.mkdir(
    parents=True,
    exist_ok=True
)

# 自动选择训练设备
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
