"""在同一测试集上比较 FP32 与 INT8 分类质量，评估时不重新量化。"""

import argparse
from dataclasses import replace
import gc
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import config
from dataset import build_dataloader
from quantize_model import load_int8_model as load_saved_int8_model


def load_fp32_model():
    """从本地已微调目录加载 FP32 模型，并关闭 Dropout。"""
    model = AutoModelForSequenceClassification.from_pretrained(config.fp32_model_dir, local_files_only=True)
    return model.cpu().float().eval()


# 同时兼容早期代码中的 load_fp32 名称和当前教学版的 load_fp32_model。
load_fp32 = load_fp32_model


def load_int8_model():
    """恢复保存的量化参数；不复制 FP32 权重，也不执行第二次量化。"""
    if not Path(config.int8_state_path).is_file():
        raise FileNotFoundError('找不到 INT8 权重，请先运行 quantize_model.py。')
    # 保存格式是 TorchAO INT8 state_dict，复用 weights_only=True、assign=True 加载。
    return load_saved_int8_model(config.int8_state_path)


def evaluate_model(model, dataloader, device='cpu'):
    """汇总所有样本后计算指标；不能简单平均大小不同批次的准确率。"""
    device = torch.device(device)
    if device.type != 'cpu':
        raise ValueError('当前量化对比实验使用 CPU。')
    model = model.to(device).eval()
    num_labels = model.config.num_labels
    y_true, y_pred = [], []
    # eval 关闭 Dropout；inference_mode 关闭梯度及其记录，二者作用不同。
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch['labels']
            if labels.ndim != 1 or labels.dtype != torch.long:
                raise ValueError('labels 必须是一维 torch.long 张量。')
            if not ((labels >= 0) & (labels < num_labels)).all():
                raise ValueError(f'标签超出有效范围 0 到 {num_labels - 1}。')
            # labels 留作指标计算，不传给模型计算损失，也不从原始 batch 删除。
            inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
            logits = model(**inputs).logits
            if logits.shape != (len(labels), num_labels) or not torch.isfinite(logits).all():
                raise ValueError('模型输出形状错误，或包含 NaN/Inf。')
            # 类别维度为最后一维；无需 softmax，最大值对应的下标不变。
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
    if not y_true:
        raise ValueError('没有有效评估样本。')
    # 宏平均固定覆盖全部类别；无样本或未预测到的类别，零分母按 0 处理。
    label_ids = list(range(num_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average='macro', zero_division=0,
    )
    class_names = [str(model.config.id2label[index]) for index in label_ids]
    return {'samples': len(y_true), 'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
            'report': classification_report(y_true, y_pred, labels=label_ids,
                                            target_names=class_names, digits=4, zero_division=0)}


def run_evaluation(max_samples=None, batch_size=None):
    """用相同输入依次评估两种模型，逐个释放，减少同时占用的内存。"""
    if torch.device(config.device).type != 'cpu':
        raise ValueError('本阶段请在 config.py 中设置 device="cpu"。')
    current_config = config if batch_size is None else replace(config, batch_size=batch_size)
    tokenizer = AutoTokenizer.from_pretrained(config.fp32_model_dir, local_files_only=True)
    loader = build_dataloader(current_config.test_path, tokenizer, shuffle=False,
                              max_samples=max_samples, config=current_config)
    print(f'测试样本数：{len(loader.dataset)}；批大小：{current_config.batch_size}；设备：CPU', flush=True)
    if max_samples is not None:
        print('仅取文件开头的部分样本，若数据按类别排列，结果不能代表完整测试集。')
    results = {}
    reference_labels = None
    for name, load_model in [('FP32', load_fp32_model), ('INT8', load_int8_model)]:
        print(f'正在加载并评估 {name}……', flush=True)
        model = load_model()
        if reference_labels is None:
            reference_labels = dict(model.config.id2label)
        elif dict(model.config.id2label) != reference_labels:
            raise ValueError('FP32 与 INT8 类别映射不一致，请检查量化模型来源。')
        results[name] = evaluate_model(model, loader)
        metrics = results[name]
        print(f"{name}：Accuracy={metrics['accuracy']:.4f}，Precision={metrics['precision']:.4f}，"
              f"Recall={metrics['recall']:.4f}，F1={metrics['f1']:.4f}", flush=True)
        print(metrics['report'])
        del model
        gc.collect()
    print('Precision、Recall、F1 为全部类别宏平均；本脚本评估质量，不进行速度计时。')
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='在相同测试集上评估 FP32 与 INT8 模型')
    parser.add_argument('--max-samples', type=int, default=None, help='调试时取前 N 条，默认全部')
    parser.add_argument('--batch-size', type=int, default=None, help='默认使用 config.py 的批大小')
    parser.add_argument('--threads', type=int, default=None, help='可选的 CPU 计算线程数')
    args = parser.parse_args(argv)
    for name, value in [('max-samples', args.max_samples), ('batch-size', args.batch_size), ('threads', args.threads)]:
        if value is not None and value <= 0:
            parser.error(f'--{name} 必须大于 0')
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    try:
        run_evaluation(args.max_samples, args.batch_size)
    except (ValueError, OSError) as error:
        print(f'评估未完成：{error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
