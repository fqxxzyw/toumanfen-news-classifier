import fasttext
from cffi import model

from config import Config


# 创建config对象
conf = Config()

# 训练字符集

def train_char():
    train_path = conf.processed_data_dir / 'train_char.txt'
    fasttext.train_supervised(train_path)
    print("train_char训练完成")
    return model
