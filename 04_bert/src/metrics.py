"""训练与后续测试共用的单标签多分类指标。"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, recall_score, \
    classification_report, precision_score


def cal_metrics(y_true, y_pred, class_name):
    """
    计算单标签多分类指标。
    """
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)
    # 计算精确率
    precision = precision_score(y_true,y_pred,average='macro')
    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro')
    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='macro')

    # 计算分类报告
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_name))),
        target_names=class_name, zero_division=0
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification Report:\n{report}")

    return acc, precision, recall, f1, report


