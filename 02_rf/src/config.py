from pathlib import Path


class Config:
    """
    保存项目需要使用的文件路径
    """

    def __init__(self):
        PROJECT_PATH = Path(__file__).resolve().parents[1]
        self.artifacts_dir = PROJECT_PATH / 'artifacts'
        # 训练数据路径
        self.train_path = PROJECT_PATH / "data" / "train.csv"
        # 验证数据路径
        self.dev_path = PROJECT_PATH / "data" / "dev.csv"
        # 测试数据路径
        self.test_path = PROJECT_PATH / "data" / "test.csv"
        # 停用词文件路径
        self.stop_words_path = PROJECT_PATH / "data" / "stopwords.txt"
        # 类别路径
        self.class_name_path = PROJECT_PATH / "data" / "class.txt"
        # 随机森林模型保存路径
        self.rf_model_save_path = PROJECT_PATH / "models" / "rf_model.pkl"
        # TF-IDF向量化器保存路径
        self.tfidf_model_save_path = PROJECT_PATH / "models" / "tfidf_model.pkl"
