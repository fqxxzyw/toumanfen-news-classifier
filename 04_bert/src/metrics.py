"""训练和测试共用的评估函数。

适用场景：单标签多分类，即每条新闻只有一个真实类别，模型也只预测一个类别。
本文件接收类别 ID，不接收模型输出的 logits 或概率；预测 ID 由 train.py 中
的 torch.argmax(logits, dim=1) 得到，再传入这里计算指标。
"""

from sklearn.metrics import (
    accuracy_score,          # 准确率：所有样本中预测正确的比例
    precision_score,         # 精确率：预测为某类的样本中，有多少确实属于该类
    recall_score,            # 召回率：实际属于某类的样本中，有多少被成功找出
    f1_score,                # F1：综合衡量精确率和召回率
    classification_report,  # 分类报告：展示各类别的指标和样本数量
)


def cal_metrics(y_true, y_pred, class_name):
    """
    计算并打印评估指标，返回四个分数和一份分类报告。

    参数：
        y_true：真实类别 ID 序列，例如 [0, 1, 2, 1]。
        y_pred：预测类别 ID 序列，例如 [0, 2, 2, 1]。
                两个序列必须长度相同，同一位置必须对应同一个样本。
        class_name：类别名称列表，列表下标就是类别 ID。
                    例如 ['财经', '体育', '科技'] 表示 0=财经、1=体育、2=科技。
                    顺序必须与训练时 class_list 的顺序一致。

    返回值（元组，顺序固定）：
        (acc, precision, recall, f1, report)
        前四项是 0～1 之间的分数，越大越好；report 是报告字符串。
        可用 acc, precision, recall, f1, report = cal_metrics(...) 解包。
        train.py 中的 dev_metrics[3] 取出的就是这里的 f1。
    """
    # 1. 准确率（Accuracy）= 预测正确的样本数 / 总样本数。
    # 例如 100 条新闻中有 90 条分类正确，acc 就是 0.9。
    # 当各类别样本数量差异很大时，单看准确率可能忽略少数类别的表现。
    acc = accuracy_score(y_true, y_pred)

    # 以下三个指标都使用 average='macro'（宏平均）：
    # 先分别计算每个类别的指标，再对类别求算术平均，每个类别权重相同。
    # 默认参与计算的是 y_true 或 y_pred 中出现的类别；两者都未出现的类别
    # 不参与这三个宏平均。下面的分类报告则会列出 class_name 中的全部类别。
    # 多分类任务需要指定平均方式，不能使用默认的二分类计算方式。
    #
    # 为了理解单个类别的公式，可以把该类别看作“正类”，其他类别看作“负类”：
    # TP：实际属于该类，也预测成该类。
    # FP：实际不属于该类，却预测成该类（误报）。
    # FN：实际属于该类，却预测成其他类（漏报）。

    # 2. 精确率（Precision）= TP / (TP + FP)。
    # 例如预测为“体育”的 10 条新闻中，8 条真的是体育，体育类精确率为 0.8。
    # 如果某类没有任何预测样本，其精确率分母为 0；当前默认会警告并记为 0。
    precision = precision_score(y_true,y_pred,average='macro')

    # 3. 召回率（Recall）= TP / (TP + FN)。
    # 例如实际有 20 条体育新闻，找对了 16 条，体育类召回率为 0.8。
    # 精确率关注“预测出来的有多准”，召回率关注“实际存在的找出了多少”。
    recall = recall_score(y_true, y_pred, average='macro')

    # 4. 单个类别的 F1 = 2 * Precision * Recall / (Precision + Recall)。
    # 它是精确率和召回率的调和平均，其中一项很低时，F1 也会受到影响。
    # 这里的宏平均 F1 是“各类别 F1 的平均值”，不是把上面已经平均好的
    # precision 和 recall 代入公式。训练时使用这个 F1 选择最优模型。
    f1 = f1_score(y_true, y_pred, average='macro')

    # 5. 生成逐类别报告，便于观察哪个类别容易误判或漏判。
    # labels 显式列出全部类别 ID：例如有 3 类时就是 [0, 1, 2]。
    # target_names 按相同顺序提供名称，让报告显示“体育”等名称而不是数字 ID。
    # 即使某个类别在本次评估中完全没有出现，也会保留对应行。
    # zero_division=0：报告中遇到分母为 0 的指标时填 0，不发出除零警告。
    # 注意：这个选项只作用于本次 classification_report 调用。
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_name))),
        target_names=class_name, zero_division=0
    )

    # 6. 打印指标。:.4f 表示保留小数点后四位，例如 0.923456 显示为 0.9235。
    # 报告中的 support 是每个类别的真实样本数；macro avg 是各类别简单平均，
    # weighted avg 则按各类别的真实样本数加权平均。
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification Report:\n{report}")

    # 7. 返回原始数值供训练程序使用，打印时的小数位格式不会影响返回精度。
    # 顺序必须与调用方的解包顺序或下标对应，F1 位于下标 3。
    return acc, precision, recall, f1, report

