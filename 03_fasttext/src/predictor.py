import os

import fasttext

from config import Config
from preprocess import preprocess_text

conf = Config()


class FastTextPredictor:
    def __init__(self, model_path,mode='word'):
        self.model = fasttext.load_model(os.path.relpath(model_path))
        self.mode = mode

    def predict(self, text):
        pre_text = preprocess_text(text,mode=self.mode)

        # 使用批量接口，避开当前 fastText 单条预测与 NumPy 2 的兼容问题。
        labels, probs = self.model.predict([pre_text], k=1)
        label = labels[0][0].removeprefix('__label__')

        probability = float(probs[0][0])
        result={
            'label':label,
            'probability':probability
        }
        return result


if __name__ == '__main__':
    model_path = conf.model_dir / 'fasttext_char_autotune.bin'
    predictor = FastTextPredictor(model_path, mode='char')
    text = "这个电影"
    result = predictor.predict(text)
    print(result)
