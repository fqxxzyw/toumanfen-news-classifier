# 导入 Path，用于处理项目路径
from pathlib import Path

# 导入 dataclass，用于创建配置对象
from dataclasses import dataclass


# 获取项目根目录
BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

# @dataclass 可以理解成“帮我们自动写构造方法的数据类工具”，特别适合 Config、User、Student 这种主要用来保存数据的类。
@dataclass
class Config:

    # -------------------------------
    # 数据路径
    # -------------------------------

    train_path: Path = (
        BASE_DIR
        /
        "data"
        /
        "train.txt"
    )

    dev_path: Path = (
        BASE_DIR
        /
        "data"
        /
        "dev.txt"
    )

    test_path: Path = (
        BASE_DIR
        /
        "data"
        /
        "test.txt"
    )

    class_path: Path = (
        BASE_DIR
        /
        "data"
        /
        "class.txt"
    )

    output_dir: Path = (
        BASE_DIR
        /
        "outputs"
    )

    # -------------------------------
    # DeepSeek API
    # -------------------------------

    base_url: str = (
        "https://api.deepseek.com"
    )

    # 新闻分类默认优先使用 Flash
    model_name: str = (
        "deepseek-flash"
    )

    # 分类输出很短
    # 没必要允许模型生成几千 Token
    max_tokens: int = 128

    # API 请求超时时间
    timeout: float = 30.0


config = Config()


# 自动创建输出目录
config.output_dir.mkdir(
    parents=True,
    exist_ok=True
)