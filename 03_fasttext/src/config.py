# 导入 Path，用来更加规范地处理文件路径
from pathlib import Path
# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    def __init__(self):
        # 原始数据目录
        self.raw_data_dir = (
            BASE_DIR / 'data' / 'raw'
        )


        # 处理后的数据目录
        self.processed_data_dir = (
            BASE_DIR / "data" / "processed"
        )

        # 模型保存目录
        self.model_dir = (
            BASE_DIR / "models"
        )

        # 训练集
        self.train_path = (
            self.raw_data_dir / "train.txt"
        )

        # 验证集
        self.dev_path = (
            self.raw_data_dir / "dev.txt"
        )

        # 测试集
        self.test_path = (
            self.raw_data_dir / "test.txt"
        )

        # 类别文件
        self.class_path = (
            self.raw_data_dir / "class.txt"
        )

        # 自动创建目录
        self.processed_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":

    # 创建配置对象
    config = Config()

    # 打印关键路径
    print("项目根目录：", BASE_DIR)
    print("训练集：", config.train_path)
    print("模型目录：", config.model_dir)