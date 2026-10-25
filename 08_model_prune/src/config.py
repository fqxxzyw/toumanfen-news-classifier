# 导入 dataclass
from dataclasses import dataclass
# 导入 Path
from pathlib import Path
# 导入 PyTorch
import torch

# 获取 pruning_project 根目录
BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


@dataclass
class Config:
    # 剪枝前已经训练好的 BERT 分类模型
    dense_model_path: Path = (
        BASE_DIR
        / "models"
        / "original"
    )

    # 剪枝 + 恢复训练后的模型保存目录
    pruned_model_path: Path = (
        BASE_DIR
        / "models"
        / "pruned"
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

    # 类别名称路径
    class_path: Path = (
        BASE_DIR
        / "data"
        / "class.txt"
    )

    # 输出目录
    output_dir: Path = (
        BASE_DIR
        / "outputs"
    )

    # 类别数
    num_classes: int = 10

    # 最大文本长度
    max_length: int = 128

    # Batch 大小
    batch_size: int = 16

    # 剪枝比例
    prune_amount: float = 0.3

    # 剪枝后的恢复训练轮数
    recovery_epochs: int = 3

    # 恢复训练学习率
    learning_rate: float = 2e-5

    # DataLoader 子进程数量
    num_workers: int = 0

    # Benchmark 预热次数
    benchmark_warmup: int = 10

    # Benchmark 正式测试次数
    benchmark_runs: int = 50


# 创建配置对象
config = Config()

# 自动创建剪枝模型保存目录
config.pruned_model_path.mkdir(
    parents=True,
    exist_ok=True
)

# 自动创建输出目录
config.output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# 自动选择设备
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
