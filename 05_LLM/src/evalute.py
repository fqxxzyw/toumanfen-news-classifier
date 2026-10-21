"""离线评估已有预测 CSV，不调用 API，也不修改原始预测文件。"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from config import config


def _to_label_ids(values, class_names, column_name):
    """统一英文类别和数字 ID：例如 sports 与 7 都转换成整数 7。"""
    name_to_id = {name: index for index, name in enumerate(class_names)}
    text = values.astype(str).str.strip()
    # 先映射英文类别；未映射的内容再尝试按数字 ID 解析。
    ids = text.map(name_to_id).fillna(pd.to_numeric(text, errors='coerce'))
    valid = ids.notna() & ids.mod(1).eq(0) & ids.between(0, len(class_names) - 1)
    if not valid.all():
        raise ValueError(f'{column_name} 中有 {int((~valid).sum())} 个空值或非法类别，请检查预测 CSV。')
    return ids.astype(int)


def evaluate_predictions(csv_path=None, class_path=None):
    """评估每个 row_id 的最后一次成功预测，返回指标、分类报告和样本统计。"""
    csv_path = config.output_dir / 'dev_predictions.csv' if csv_path is None else Path(csv_path)
    class_path = config.class_path if class_path is None else Path(class_path)

    # 标签 ID 由 class.txt 的行号决定，不能依赖另一个字典碰巧相同的排列顺序。
    class_names = [name.strip() for name in class_path.read_text(encoding='utf-8-sig').splitlines()]
    if not class_names or any(not name for name in class_names) or len(set(class_names)) != len(class_names):
        raise ValueError('class.txt 不能为空、包含空类别或重复类别。')

    # 全部按字符串读取，保留原值；utf-8-sig 同时兼容带 BOM 与不带 BOM 的文件。
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as error:
        raise ValueError('预测 CSV 为空，请先运行批量预测。') from error
    missing = {'row_id', 'status', 'pred_label'} - set(df.columns)
    if missing:
        raise ValueError(f'预测 CSV 缺少必要列：{", ".join(sorted(missing))}。')

    # 兼容当前旧文件的拼写 true_lable，以及新格式 true_id / true_label。
    # 不直接改写旧文件，已有 API 预测结果可以继续评估。
    true_columns = [name for name in ('true_id', 'true_label', 'true_lable') if name in df.columns]
    if not true_columns:
        raise ValueError('预测 CSV 缺少真实标签列：需要 true_id、true_label 或旧拼写 true_lable。')
    if df.empty:
        raise ValueError('预测 CSV 只有表头，没有可评估的数据。')
    row_ids = pd.to_numeric(df['row_id'], errors='coerce')
    if not (row_ids.notna() & row_ids.mod(1).eq(0) & row_ids.ge(0)).all():
        raise ValueError('row_id 必须是非负整数。')
    df['row_id'] = row_ids.astype(int)
    df['status'] = df['status'].str.strip()
    if not df['status'].isin(['success', 'fail', 'failed']).all():
        raise ValueError('status 只能是 success、fail 或 failed。')

    # 一条新闻可能先失败、后成功，或者保存过多次成功记录。
    # 先筛选成功记录，再按 row_id 保留最后一次成功，避免重试次数影响指标。
    success_df = df.loc[df['status'] == 'success'].drop_duplicates('row_id', keep='last').copy()
    if success_df.empty:
        raise ValueError('没有 status=success 的预测记录，暂时无法计算分类指标；请先检查 error 列。')

    y_true = _to_label_ids(success_df[true_columns[0]], class_names, true_columns[0])
    # 新格式同时有真实 ID 和英文名称时，检查两者一致，避免悄悄使用错误的标签。
    for column in true_columns[1:]:
        if not y_true.equals(_to_label_ids(success_df[column], class_names, column)):
            raise ValueError('CSV 中的多个真实标签列不一致，请检查类别顺序或标签内容。')
    y_pred = _to_label_ids(success_df['pred_label'], class_names, 'pred_label')

    # 明确指定所有类别：macro 指标是 class.txt 中全部类别指标的算术平均。
    # 某类没有样本或没有被预测到时，相关分母为 0 的指标按 0 处理。
    # 小样本不覆盖全部类别时，macro F1 也会受影响，需结合各类 support 阅读。
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0,
    )
    total = int(df['row_id'].nunique())
    evaluated = len(success_df)
    return {
        'total': total,
        'evaluated': evaluated,
        'unsuccessful': total - evaluated,
        'success_rate': evaluated / total,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'report': classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0,
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='评估已有新闻分类 CSV，不调用 API')
    parser.add_argument('--input', type=Path, default=config.output_dir / 'dev_predictions.csv',
                        help='待评估的预测 CSV')
    args = parser.parse_args(argv)
    try:
        result = evaluate_predictions(args.input)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f'评估未完成：{error}')
        return 1

    print(f"结果文件中的新闻数（按 row_id 去重）：{result['total']}")
    print(f"成功并参与评估：{result['evaluated']}；尚未成功：{result['unsuccessful']}")
    print(f"预测成功率：{result['success_rate']:.2%}")
    print('以下分类指标仅针对成功样本，macro 对 class.txt 中的全部类别取平均：')
    print(f"准确率 Accuracy：{result['accuracy']:.4f}")
    print(f"宏平均精确率 Precision：{result['precision']:.4f}")
    print(f"宏平均召回率 Recall：{result['recall']:.4f}")
    print(f"宏平均 F1：{result['f1']:.4f}")
    print('\n各类别报告（support 为该类实际样本数）：')
    print(result['report'])
    return 0


# 导入时仅定义函数，直接运行此文件时才读取 CSV 并输出评估结果。
if __name__ == '__main__':
    raise SystemExit(main())
