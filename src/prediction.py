"""供接口调用的预测模块，不作为编号步骤单独启动。"""
import pandas as pd
import numpy as np
import pickle
import config
import jieba
conf = config.Config()


# 1.加载rf_model.pkl和tfidf_model.pkl
rf_model = pickle.load(open(conf.rf_model_save_path, 'rb'))
tfidf_model = pickle.load(open(conf.tfidf_model_save_path, 'rb'))

def predict_fun(data):
    # 1.分词
    word = ' '.join(jieba.lcut(data['text'])[:30])
    # 2.词向量化
    x = tfidf_model.transform([word])
    # 3.预测
    pred = rf_model.predict(x)[0]
    print('预测结果：',pred)
    # 4.设置标签索引和名称对应的关系{0:'finance',....}
    id2class = {i:line.strip() for i,line in enumerate(open(conf.class_name_path,encoding='utf-8'))}
    print("预测结果:",id2class[pred])
    data['label'] = id2class[pred]
    return data
if __name__ == '__main__':
    predict_fun({'text': '中国队在世界锦标赛中夺得冠军'})
