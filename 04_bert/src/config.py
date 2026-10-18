# 导入 Path，用于跨平台处理文件路径
from pathlib import Path

# 导入 torch，用于判断训练设备
import torch


# 当前文件位于 src/config.py
# parent.parent 可以得到项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:

    def __init__(self):

        # =====================================
        # 1. 数据路径
        # =====================================

        # 数据目录
        self.data_dir = BASE_DIR / "data"

        # 训练集
        self.train_path = self.data_dir / "train.txt"

        # 验证集
        self.dev_path = self.data_dir / "dev.txt"

        # 测试集
        self.test_path = self.data_dir / "test.txt"

        # 类别文件
        self.class_path = self.data_dir / "class.txt"

        # =====================================
        # 2. 预训练模型路径
        # =====================================

        # 本地 bert-base-chinese 目录
        self.pretrained_model_path = (
                BASE_DIR
                /
                "pretrained"
                /
                "bert-base-chinese"
        )

        # =====================================
        # 3. 模型输出目录
        # =====================================

        # 最佳模型保存目录
        self.best_model_dir = (
                BASE_DIR
                /
                "models"
                /
                "best_model"
        )

        # 自动创建模型目录
        self.best_model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =====================================
        # 4. 训练超参数
        # =====================================

        # 每个 Batch 的样本数
        self.batch_size = 16

        # 最大 Token 长度
        self.max_length = 64

        # 最大训练轮数
        self.num_epochs = 5

        # BERT 微调常用较小学习率
        self.learning_rate = 2e-5

        # 权重衰减
        self.weight_decay = 0.01

        # Warmup 比例
        self.warmup_ratio = 0.1

        # Early Stopping 容忍轮数
        self.patience = 2

        # 梯度裁剪阈值
        self.max_grad_norm = 1.0

        # =====================================
        # 5. 其他配置
        # =====================================

        # 随机种子
        self.seed = 42

        # Windows 教学环境建议从 0 开始
        self.num_workers = 0

        # 自动选择 GPU 或 CPU
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


# 创建全局配置对象
conf= Config()
