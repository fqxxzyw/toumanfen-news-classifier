# 导入 PyTorch
import torch
# 导入 AdamW 优化器
from torch.optim import AdamW
# 导入 tqdm
from tqdm import tqdm

# 导入配置
from config import config, device
# 导入 DataLoader 构建函数
from dataset import build_dataloader
# 导入 Teacher 加载函数
from teacher import load_teacher
# 导入 Student
from student import StudentBiLSTM
# 导入统一蒸馏 Loss
from distillation_loss import distillation_loss
# 导入评估函数
from evaluate import evaluate


def _load_teacher_cache():
    # 判断两个缓存文件是否都存在
    cache_exists = (
        config.teacher_logits_cache.exists()
        and
        config.teacher_hidden_cache.exists()
    )

    # 如果关闭缓存功能或缓存不存在，就返回 None
    if (
        not config.use_teacher_cache
        or
        not cache_exists
    ):
        return (
            None,
            None
        )

    # 从 CPU 加载 Teacher Logits
    teacher_logits_cache = torch.load(
        config.teacher_logits_cache,
        map_location="cpu"
    )

    # 从 CPU 加载 Teacher Hidden
    teacher_hidden_cache = torch.load(
        config.teacher_hidden_cache,
        map_location="cpu"
    )

    # 返回两个缓存 Tensor
    return (
        teacher_logits_cache,
        teacher_hidden_cache
    )


def train_distillation(
    teacher,
    student,
    train_loader,
    dev_loader,
    teacher_logits_cache=None,
    teacher_hidden_cache=None
):
    # Teacher 始终保持 eval 模式
    teacher.eval()

    # 把 Teacher 放到设备
    teacher.to(
        device
    )

    # 把 Student 放到设备
    student.to(
        device
    )

    # 优化器只管理 Student 参数
    optimizer = AdamW(
        student.parameters(),
        lr=config.learning_rate
    )

    # 记录最佳验证集 Macro-F1
    best_f1 = -1.0

    # 记录连续多少轮没有提升
    no_improve = 0

    # 开始 Epoch 循环
    for epoch in range(config.epochs):
        # Student 切换到训练模式
        student.train()

        # 记录当前 Epoch 总 Loss
        total_loss = 0.0

        # 遍历训练集 Batch
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}"
        ):
            # 把输入 Token ID 放到设备
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

            # 把真实标签放到设备
            labels = batch[
                "labels"
            ].to(
                device
            )

            # 读取当前样本在数据集中的原始下标
            indices = batch[
                "indices"
            ]

            # 清空 Student 上一次的梯度
            optimizer.zero_grad()

            # 如果 Teacher 缓存存在，就直接按下标读取
            if (
                teacher_logits_cache is not None
                and
                teacher_hidden_cache is not None
            ):
                # 读取当前 Batch 对应的 Teacher Logits
                teacher_logits = teacher_logits_cache[
                    indices
                ].to(
                    device
                )

                # 读取当前 Batch 对应的 Teacher Hidden
                teacher_hidden = teacher_hidden_cache[
                    indices
                ].to(
                    device
                )

            # 如果没有缓存，就在线运行 Teacher
            else:
                # Teacher 不计算梯度
                with torch.inference_mode():
                    # Teacher 前向
                    teacher_outputs = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )

                    # 获取 Teacher Logits
                    teacher_logits = teacher_outputs.logits

                    # 获取 Teacher 最后一层 [CLS] Hidden
                    teacher_hidden = teacher_outputs.hidden_states[-1][
                        :,
                        0,
                        :
                    ]

            # Student 前向
            student_logits, student_hidden = student(
                input_ids,
                attention_mask
            )

            # 计算蒸馏 Loss
            losses = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                temperature=config.temperature,
                hard_weight=config.hard_weight,
                soft_weight=config.soft_weight,
                feature_weight=config.feature_weight
            )

            # 取出总 Loss
            loss = losses[
                "loss"
            ]

            # 反向传播
            loss.backward()

            # 更新 Student 参数
            optimizer.step()

            # 累加当前 Batch Loss
            total_loss += loss.item()

        # 每个 Epoch 结束后在 Dev 集评估
        dev_result = evaluate(
            student,
            dev_loader,
            device
        )

        # 取出 Dev Macro-F1
        dev_f1 = dev_result[
            "macro_f1"
        ]

        # 输出当前训练结果
        print(
            f"Epoch={epoch + 1}, "
            f"Loss={total_loss / max(1, len(train_loader)):.4f}, "
            f"Dev Accuracy={dev_result['accuracy']:.4f}, "
            f"Dev Macro-F1={dev_f1:.4f}"
        )

        # 如果本轮更好，就保存 Student
        if dev_f1 > best_f1:
            # 更新最佳 F1
            best_f1 = dev_f1

            # 保存 Student 参数
            torch.save(
                student.state_dict(),
                config.student_path
            )

            # Early Stop 计数清零
            no_improve = 0

        # 如果没有提升
        else:
            # 未提升次数加 1
            no_improve += 1

            # 达到容忍轮数时停止
            if no_improve >= config.patience:
                # 输出停止提示
                print(
                    "Early Stopping"
                )

                # 结束 Epoch 循环
                break


def main():
    # 加载 Teacher Tokenizer 和模型
    tokenizer, teacher = load_teacher(
        config.teacher_path,
        config.num_classes
    )

    # 创建训练集 DataLoader
    train_loader = build_dataloader(
        config.train_path,
        tokenizer,
        config.max_length,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # 创建验证集 DataLoader
    dev_loader = build_dataloader(
        config.dev_path,
        tokenizer,
        config.max_length,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # 获取 Teacher 隐藏维度
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
    )

    # 尝试读取 Teacher 离线缓存
    teacher_logits_cache, teacher_hidden_cache = _load_teacher_cache()

    # 输出当前使用的 Teacher 模式
    if teacher_logits_cache is not None:
        # 告诉用户当前使用离线缓存
        print(
            "当前模式：使用 Teacher 离线缓存"
        )
    else:
        # 告诉用户当前在线运行 Teacher
        print(
            "当前模式：Teacher 在线前向"
        )

    # 开始蒸馏训练
    train_distillation(
        teacher,
        student,
        train_loader,
        dev_loader,
        teacher_logits_cache,
        teacher_hidden_cache
    )


# 直接运行当前文件时执行 main
if __name__ == "__main__":
    main()
