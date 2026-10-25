# 导入 PyTorch
import torch
# 导入 Accuracy
from sklearn.metrics import accuracy_score
# 导入 Macro-F1
from sklearn.metrics import f1_score


def evaluate(
    model,
    data_loader,
    device
):
    # 切换到评估模式
    model.eval()

    # 保存真实标签
    all_labels = []

    # 保存预测标签
    all_preds = []

    # 推理不计算梯度
    with torch.inference_mode():
        # 遍历所有 Batch
        for batch in data_loader:
            # 把输入放到设备
            input_ids = batch[
                "input_ids"
            ].to(
                device
            )

            # 把 Mask 放到设备
            attention_mask = batch[
                "attention_mask"
            ].to(
                device
            )

            # 保留 CPU 版真实标签
            labels = batch[
                "labels"
            ]

            # 模型前向
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # 取分类 Logits
            logits = outputs.logits

            # 取最大类别
            preds = torch.argmax(
                logits,
                dim=-1
            )

            # 保存真实标签
            all_labels.extend(
                labels.tolist()
            )

            # 保存预测标签
            all_preds.extend(
                preds.cpu().tolist()
            )

    # 计算 Accuracy
    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    # 计算 Macro-F1
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    # 返回指标
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }
