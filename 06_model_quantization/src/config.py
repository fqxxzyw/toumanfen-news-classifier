from dataclasses import dataclass

from pathlib import Path

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

@dataclass
class Config:
    fp32_model_dir: Path = (
        # 指向直接包含 config.json 和权重文件的目录，而不是外层容器目录。
        BASE_DIR / 'models' / 'bert_fp32' / 'best_model'
    )
    # INT8 state_dict 文件路径；父目录另存配置和分词器。
    int8_state_path: Path = (
        BASE_DIR
        /
        "models"
        /
        "bert_int8"
        /
        "model_state.pt"
    )

    # 测试集路径
    test_path: Path = (
            BASE_DIR
            /
            "data"
            /
            "test.txt"
    )

    # 动态量化实验使用 CPU
    device: str = "cpu"

    # 每次取多少条样本
    batch_size: int = 32

    # BERT 最大文本长度
    max_length: int = 128

    # 正式计时前先预热多少次
    warmup_steps: int = 20

    # 正式重复测试多少次
    benchmark_steps: int = 100

    # =========================================================
    # 3. 创建配置对象
    # =========================================================
config = Config()

config.int8_state_path.parent.mkdir(parents=True, exist_ok=True)
