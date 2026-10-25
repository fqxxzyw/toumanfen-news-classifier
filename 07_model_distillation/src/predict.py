# 导入 PyTorch
import torch

# 导入配置
from config import config, device
# 导入类别名称加载函数
from dataset import load_classes
# 导入 Teacher，仅使用其 Tokenizer 和配置
from teacher import load_teacher
# 导入 Student
from student import StudentBiLSTM


def load_student():
    # 加载 Teacher Tokenizer 和 Teacher 配置
    tokenizer, teacher = load_teacher(
        config.teacher_path,
        config.num_classes
    )

    # 获取 Teacher Hidden 维度
    teacher_hidden_size = getattr(
        teacher.config,
        "hidden_size",
        768
    )
    # 创建 Student 结构
    student = StudentBiLSTM(
        vocab_size=len(tokenizer),
        num_classes=config.num_classes,
        teacher_hidden_state=teacher_hidden_size,
        embedding_dim=config.embed_size,
        hidden_dim=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)

    # 加载 Student 参数
    student.load_state_dict(
        torch.load(
            config.student_path,
            map_location=device
        )
    )

    # 切换到评估模式
    student.eval()

    # 返回 Tokenizer 和 Student
    return tokenizer, student


def predict(
        texts
):
    # 加载模型
    tokenizer, student = load_student()

    # 批量 Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    )

    # 不计算梯度
    with torch.inference_mode():
        # Student 前向
        logits, _ = student(
            encoded[
                "input_ids"
            ].to(
                device
            ),
            encoded[
                "attention_mask"
            ].to(
                device
            )
        )

        # 转成类别概率
        probs = torch.softmax(
            logits,
            dim=-1
        )

        # 找到概率最大类别
        preds = torch.argmax(
            probs,
            dim=-1
        )

    # 加载类别名称
    classes = load_classes(
        config.class_path
    )

    # 创建结果列表
    results = []

    # 遍历每一条文本
    for i, text in enumerate(
            texts
    ):
        # 获取预测类别编号
        pred_id = int(
            preds[i].item()
        )

        # 如果有 class.txt，就显示类别名称
        pred_name = (
            classes[pred_id]
            if pred_id < len(classes)
            else str(pred_id)
        )
        print(probs[i].max())
        # # 获取最大类别概率
        confidence = probs[i].max().item()

        # 保存预测结果
        results.append(
            {
                "text": text,
                "label_id": pred_id,
                "label_name": pred_name,
                "confidence": confidence
            }
        )

    # 返回结果
    return results


def main():
    # 准备两条演示文本
    texts = [
        "球队在决赛中逆转夺冠",
        "科技公司发布新一代人工智能芯片"
    ]

    # 执行预测
    results = predict(
        texts
    )

    # 逐条打印结果
    for item in results:
        # 输出结果字典
        print(
            item
        )


# 直接运行当前文件时执行 main
if __name__ == "__main__":
    main()