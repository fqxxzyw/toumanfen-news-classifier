# 导入 PyTorch
import torch
# 导入分类指标
from sklearn.metrics import accuracy_score
# 导入 Macro-F1
from sklearn.metrics import f1_score

# 导入配置
from config import config, device
# 导入数据加载器
from dataset import build_dataloader
# 导入 Teacher 加载函数，只使用它的 Tokenizer
from teacher import load_teacher
# 导入 Student
from student import StudentBiLSTM


def evaluate(
    model,
    data_loader,
    device
):
    # 切换到评估模式
    model.eval()

    # 保存所有真实标签
    all_labels = []

    # 保存所有预测标签
    all_preds = []

    # 推理阶段不计算梯度
    with torch.inference_mode():
        # 遍历所有 Batch
        for batch in data_loader:
            # 把 input_ids 放到设备
            input_ids = batch[
                "input_ids"
            ].to(
                device
            )

            # 把 attention_mask 放到设备
            attention_mask = batch[
                "attention_mask"
            ].to(
                device
            )

            # 获取真实标签
            labels = batch[
                "labels"
            ]

            # Student 前向
            logits, _ = model(
                input_ids,
                attention_mask
            )

            # 找到每条样本概率最大的类别
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

    # 返回评估结果
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


def main():
    # 加载 Teacher Tokenizer
    tokenizer, teacher = load_teacher(
        config.teacher_path,
        config.num_classes
    )

    # 读取 Teacher 隐藏维度
    teacher_hidden_size = getattr(
        teacher.config,
        "hidden_size",
        768
    )

    # 创建 Student
    student = StudentBiLSTM(
        vocab_size=len(tokenizer),
        num_classes=config.num_classes,
        teacher_hidden_state=teacher_hidden_size,
        embedding_dim=config.embed_size,
        hidden_dim=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(
        device
    )

    # 加载已经训练好的 Student 参数
    student.load_state_dict(
        torch.load(
            config.student_path,
            map_location=device
        )
    )

    # 创建测试集 DataLoader
    test_loader = build_dataloader(
        config.test_path,
        tokenizer,
        config.max_length,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # 评估 Student
    result = evaluate(
        student,
        test_loader,
        device
    )

    # 输出 Accuracy
    print(
        f"Test Accuracy = {result['accuracy']:.4f}"
    )

    # 输出 Macro-F1
    print(
        f"Test Macro-F1 = {result['macro_f1']:.4f}"
    )


# 直接运行当前文件时执行 main
if __name__ == "__main__":
    main()
