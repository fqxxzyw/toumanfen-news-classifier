import os

import fasttext

from sklearn.metrics import accuracy_score, classification_report

from config import Config

conf = Config()


def evaluate_model(model_path, dev_path):
    model = fasttext.load_model(os.path.relpath(model_path))

    y_true = []
    texts = []

    with open(dev_path, 'r', encoding='utf-8') as dev_file:
        for line in dev_file:
            line = line.strip()
            if not line:
                continue
            label, text = line.split(' ',1)
            y_true.append(label)
            texts.append(text)

    # 批量预测，避开当前 fastText 单条预测接口与 NumPy 2 的兼容问题。
    pred_labels, _ = model.predict(texts, k=1)
    y_pred = [labels[0] for labels in pred_labels]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == '__main__':
    for model_name, mode in (
        ('fasttext_char', 'char'),
        ('fasttext_word', 'word'),
        ('fasttext_char_autotune', 'char'),
        ('fasttext_word_autotune', 'word'),
    ):
        print(f"\n{'=' * 20} {model_name} {'=' * 20}")
        model_path = conf.model_dir / f'{model_name}.bin'
        dev_path = conf.processed_data_dir / f'dev_{mode}.txt'
        evaluate_model(model_path, dev_path)
