# 导入 PyTorch
import torch
# 导入函数式 API
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    student_hidden=None,
    teacher_hidden=None,
    temperature=2.0,
    hard_weight=0.4,
    soft_weight=0.5,
    feature_weight=0.1
):
    # 计算 Hard Loss：Student 学真实人工标签
    hard_loss = F.cross_entropy(
        student_logits,
        labels
    )

    # 计算 Teacher 的软概率
    teacher_probs = F.softmax(
        teacher_logits
        /
        temperature,
        dim=-1
    )

    # 计算 Student 的对数软概率
    student_log_probs = F.log_softmax(
        student_logits
        /
        temperature,
        dim=-1
    )

    # 使用 KL 散度比较 Student 和 Teacher 的概率分布
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    # 进行温度补偿
    soft_loss = (
        soft_loss
        *
        temperature
        *
        temperature
    )

    # 默认 Feature Loss 为 0
    feature_loss = torch.tensor(
        0.0,
        device=student_logits.device
    )

    # 同时传入 Teacher / Student Hidden 时才计算中间层损失
    if (
        student_hidden is not None
        and
        teacher_hidden is not None
        and
        feature_weight > 0
    ):
        # 使用 MSE 对齐隐藏表示
        feature_loss = F.mse_loss(
            student_hidden,
            teacher_hidden
        )

    # 三种 Loss 加权求和
    total_loss = (
        hard_weight
        *
        hard_loss
        +
        soft_weight
        *
        soft_loss
        +
        feature_weight
        *
        feature_loss
    )

    # 返回总 Loss 和三个子 Loss
    return {
        "loss": total_loss,
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "feature_loss": feature_loss
    }