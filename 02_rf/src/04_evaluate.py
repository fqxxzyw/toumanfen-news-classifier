"""步骤 04：在独立验证集上评估已保存的模型。"""
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from training_utils import predict_batches
import pandas as pd
import numpy as np
import pickle
import config

conf = config.Config()
# 1.加载rf_model.pkl和tfidf_model.pkl
rf_model = pickle.load(open(conf.rf_model_save_path, 'rb'))
tfidf_model = pickle.load(open(conf.tfidf_model_save_path, 'rb'))

# 2.加载数据
dev_data = pd.read_csv(conf.dev_path)
# dev 中的words
dev_words = dev_data['words'].fillna('')
# 将words转换为tfidf向量
dev_words_tfidf = tfidf_model.transform(dev_words)
# 使用rf模型进行预测
dev_pred = predict_batches(rf_model, dev_words_tfidf)
# 保存预测结果
dev_data['pred'] = dev_pred
print("dev数据预测完成")
print(dev_data[['text','label','pred']].head())

metrics = {
    'dataset': 'dev',
    'samples': len(dev_data),
    'accuracy': accuracy_score(dev_data['label'], dev_pred),
    'precision_macro': precision_score(dev_data['label'], dev_pred, average='macro', zero_division=0),
    'recall_macro': recall_score(dev_data['label'], dev_pred, average='macro', zero_division=0),
    'f1_macro': f1_score(dev_data['label'], dev_pred, average='macro', zero_division=0),
}
output_dir = conf.artifacts_dir / 'evaluation'
output_dir.mkdir(parents=True, exist_ok=True)
dev_data.to_csv(output_dir / 'dev_predictions.csv', index=False, encoding='utf-8')
(output_dir / 'dev_metrics.json').write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8',
)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print('评估结果已保存至：', output_dir)
