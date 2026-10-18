import os

import fasttext

from config import Config


# 创建config对象
conf = Config()

# 训练字符集

def train_char():
    train_path = conf.processed_data_dir / 'train_char.txt'
    # 当前 Windows 版 fastText 无法打开含中文的绝对路径，改用相对路径。
    model = fasttext.train_supervised(input=os.path.relpath(train_path))
    print("train_char训练完成")
    model_name = 'fasttext_char'
    save_model(model,model_name)

def train_word():
    train_path = conf.processed_data_dir / 'train_word.txt'
    model = fasttext.train_supervised(input=os.path.relpath(train_path))
    print("train_word训练完成")
    model_name = 'fasttext_word'
    save_model(model, model_name)

def auto_param_tuning(mode='char',duration=60):
    train_path = conf.processed_data_dir / f'train_{mode}.txt'
    dev_path = conf.processed_data_dir / f'dev_{mode}.txt'
    model = fasttext.train_supervised(
        input=os.path.relpath(train_path),
        autotuneValidationFile = os.path.relpath(dev_path),
        autotuneDuration = duration,
        verbose = 3
    )
    model_name = f'fasttext_{mode}_autotune'
    save_model(model, model_name)



def save_model(model,model_name):
    model.save_model(os.path.relpath(conf.model_dir / f'{model_name}.bin'))
    print("模型保存完成")


if __name__ == '__main__':
    train_char()
    train_word()
    auto_param_tuning(mode='char',duration=300)
    auto_param_tuning(mode='word',duration=300)
