"""步骤 03：TF-IDF 特征提取、随机森林分批训练和保存。"""
# 导包
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle # 保存模型和向量化器
from sklearn.feature_extraction.text import TfidfVectorizer # 将文本转换为TF-IDF特征
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

from config import Config
from training_utils import train_forest, predict_batches

# 可调参数：进度条在每批完成后更新；并行数越大，内存占用通常越高。
TOTAL_TREES = 100
TREES_PER_BATCH = 5
N_JOBS = 2
PREDICT_BATCH_SIZE = 2000


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description='TF-IDF + random forest training')
    parser.add_argument('--limit', type=positive_int, default=None, help='optional sample limit for a quick run')
    parser.add_argument('--trees', type=positive_int, default=TOTAL_TREES)
    parser.add_argument('--trees-per-batch', type=positive_int, default=TREES_PER_BATCH)
    parser.add_argument('--jobs', type=positive_int, default=N_JOBS)
    parser.add_argument('--predict-batch-size', type=positive_int, default=PREDICT_BATCH_SIZE)
    parser.add_argument('--output-dir', type=Path, default=None, help='optional alternative models directory')
    args = parser.parse_args(argv)
    # 1.加载数据
    conf = Config()
    if args.output_dir is not None:
        conf.rf_model_save_path = args.output_dir / "rf_model.pkl"
        conf.tfidf_model_save_path = args.output_dir / "tfidf_model.pkl"
    # 创建配置对象
    print('[1/5] 正在读取训练数据…', flush=True)
    data = pd.read_csv(conf.train_path, nrows=args.limit) # 读取完整训练数据
    words = data['words'].fillna('') # 获取分词结果，并将缺失值替换为空字符串
    labels = data['label'] # 获取标签的结果，作为模型的输出
    print(f'读取完成：{len(data)} 条样本', flush=True)

    # print(data.head())
    # print(words[:10])
    # print(labels[:10])


    # 2.文本特征分析
    # 读取停用词列表
    with open(conf.stop_words_path, encoding='utf-8-sig') as f:
        stop_words = sorted(set(f.read().split())) # 按空白字符拆分，兼容换行和空格
    print(stop_words[:5])
    # 创建TF-IDF词向量化器
    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        tokenizer=str.split, # words 已经分词，直接按空白拆分，保留单字词
        token_pattern=None,
        lowercase=False, # 与停用词表保持一致，不额外转换大小写
        dtype=np.float32,
    )
    # 先划分原始文本，再仅在训练部分拟合，避免测试集参与词表和 IDF 学习。
    train_words,test_words,y_train,y_test = train_test_split(
        words, labels, test_size=0.2, random_state=42, stratify=labels,
    )
    print('[2/5] 正在提取 TF-IDF 特征，此阶段需要等待…', flush=True)
    x_train = tfidf.fit_transform(train_words)
    x_test = tfidf.transform(test_words)
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

    # 3.模型的训练和评估
    '''
    随机森林算法：
    思想：随机森林是一种集成学习算法，它通过构建多个决策树来提高模型的性能。每个决策树都是一个二叉树，每个节点都有一个特征和一个阈值。当数据到达一个节点时，算法会根据特征的值来决定是沿着左子树还是右子树继续
    原理：
    1.随机选择样本
    2.随机选择特征
    3.构建决策树
    4.递归构建决策树
    5.投票选择结果
    6.预测结果
    7.评估模型
    8.调整参数
    9.重复步骤1-8，直到模型达到最佳性能
    10.使用模型进行预测
    api: RandomForestClassifier
    参数：
    n_estimators：决策树的数量
    max_depth：每个决策树的最大深度
    min_samples_split：每个节点的最小样本数
    min_samples_leaf：每个叶子节点的最小样本数
    random_state：随机种子
    '''

    print('[3/5] 开始分批训练；每批完成后更新进度条…', flush=True)
    model = train_forest(
        x_train, y_train, total_trees=args.trees,
        trees_per_batch=args.trees_per_batch, n_jobs=args.jobs,
    )
    # 模型预测
    print('[4/5] 开始分批预测和评估…', flush=True)
    y_pred = predict_batches(model, x_test, batch_size=args.predict_batch_size)
    print(y_pred[:10])
    # 模型评估
    '''
    average : 表示报告中的平均值的计算方式
    micro : 汇总所有类别的预测计数后计算指标
    macro : 分别计算各类别的指标，再取算术平均
    weighted : 表示报告中的加权平均值的计算方式
    samples : 对每个样本计算指标后取平均，仅适用于多标签任务
    '''

    print('准确率：',accuracy_score(y_test,y_pred))
    print('精确率（宏平均）：',precision_score(y_test,y_pred,average='macro',zero_division=0))
    print('召回率（宏平均）：',recall_score(y_test,y_pred,average='macro',zero_division=0))
    print('F1分数（宏平均）：',f1_score(y_test,y_pred,average='macro',zero_division=0))

    # 4.保存模型
    print('[5/5] 正在保存模型和向量化器…', flush=True)
    conf.rf_model_save_path.parent.mkdir(parents=True, exist_ok=True)
    conf.tfidf_model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conf.rf_model_save_path,'wb') as f:
        pickle.dump(model,f)
    with open(conf.tfidf_model_save_path,'wb') as f:
        pickle.dump(tfidf,f)
    print('模型保存成功！')




if __name__ == '__main__':
    main()
