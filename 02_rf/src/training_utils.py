"""随机森林分批增加树，以及分批预测的进度显示。"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm


def train_forest(x_train, y_train, total_trees=100, trees_per_batch=5, n_jobs=2):
    """每批在相同数据上增加决策树；保留已训练的树，不重复训练。"""
    if total_trees < 1 or trees_per_batch < 1:
        raise ValueError('树的总数和每批树数必须大于 0')
    model = RandomForestClassifier(
        n_estimators=1, warm_start=True, random_state=42, n_jobs=n_jobs,
    )
    # 森林训练偏好 CSC 格式；仅转换一次，避免每批重复转换。
    x_train = x_train.tocsc()
    x_train.sort_indices()
    with tqdm(total=total_trees, desc='随机森林训练', unit='棵') as progress:
        completed = 0
        while completed < total_trees:
            target = min(completed + trees_per_batch, total_trees)
            progress.set_postfix_str(f'正在训练第 {completed + 1}～{target} 棵')
            model.set_params(n_estimators=target)
            model.fit(x_train, y_train)
            progress.update(target - completed)
            completed = target
    return model


def predict_batches(model, features, batch_size=2000):
    """按样本分批预测，避免整批预测期间没有进度反馈。"""
    if batch_size < 1:
        raise ValueError('预测批大小必须大于 0')
    predictions = []
    with tqdm(total=features.shape[0], desc='模型预测', unit='条') as progress:
        for start in range(0, features.shape[0], batch_size):
            batch = features[start:start + batch_size]
            predictions.append(model.predict(batch))
            progress.update(batch.shape[0])
    return np.concatenate(predictions) if predictions else np.array([], dtype=model.classes_.dtype)
