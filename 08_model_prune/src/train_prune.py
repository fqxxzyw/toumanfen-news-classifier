# 导入 copy，用来保存恢复训练期间的最佳状态
import copy
# 导入 PyTorch
import torch
# 导入交叉熵
import torch.nn.functional as F
# 导入 AdamW
from torch.optim import AdamW
# 导入 tqdm
from tqdm import tqdm
# 导入配置
from config import config, device
# 导入数据加载器
from dataset import build_dataloader
# 导入模型加载函数
from model import load_model_and_tokenizer
# 导入评估函数
from evaluate import evaluate
# 导入剪枝工具
from pruning_utils import global_prune
# 导入稀疏率计算
from pruning_utils import calculate_sparsity
# 导入 remove 工具
from pruning_utils import remove_pruning


def recovery_finetune(
    model,
    train_loader,
    dev_loader,
    parameters_to_prune
):
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    # 记录最佳 Dev Macro-F1
    best_f1 = -1.0

    # 保存最佳模型状态
    best_state = None

    # 开始恢复训练
    for epoch in range(
        config.recovery_epochs
    ):
        # 切换到训练模式
        model.train()

        # 记录当前 Epoch 总 Loss
        total_loss = 0.0

        # 遍历训练集
        for batch in tqdm(
            train_loader,
            desc=f"Recovery Epoch {epoch + 1}"
        ):
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

            # 把标签放到设备
            labels = batch[
                "labels"
            ].to(
                device
            )

            # 清空梯度
            optimizer.zero_grad()

            # 前向计算
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # 使用真实标签计算交叉熵
            loss = F.cross_entropy(
                outputs.logits,
                labels
            )

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累加 Loss
            total_loss += loss.item()

        # 在 Dev 集评估
        dev_result = evaluate(
            model,
            dev_loader,
            device
        )

        # 计算当前仍然保持的剪枝稀疏率
        sparsity = calculate_sparsity(
            parameters_to_prune
        )

        # 输出恢复训练结果
        print(
            f"Epoch={epoch + 1}, "
            f"Loss={total_loss / max(1, len(train_loader)):.4f}, "
            f"Dev Accuracy={dev_result['accuracy']:.4f}, "
            f"Dev Macro-F1={dev_result['macro_f1']:.4f}, "
            f"Query Sparsity={sparsity:.4f}"
        )

        # 如果本轮更好
        if dev_result[
            "macro_f1"
        ] > best_f1:
            # 更新最佳 F1
            best_f1 = dev_result[
                "macro_f1"
            ]

            # 深拷贝当前 state_dict
            best_state = copy.deepcopy(
                model.state_dict()
            )

    # 如果记录了最佳状态
    if best_state is not None:
        # 恢复最佳参数
        model.load_state_dict(
            best_state
        )

    # 返回恢复训练后的模型
    return model


def main():
    # 加载 Dense BERT 分类模型和 Tokenizer
    tokenizer, model = load_model_and_tokenizer(
        config.dense_model_path,
        config.num_classes
    )

    # 把模型放到设备
    model = model.to(
        device
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

    # 创建测试集 DataLoader
    test_loader = build_dataloader(
        config.test_path,
        tokenizer,
        config.max_length,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # 先评估 Dense 模型
    dense_result = evaluate(
        model,
        test_loader,
        device
    )

    # 输出剪枝前效果
    print(
        "剪枝前：",
        dense_result
    )

    # 对 12 层 Query Linear 做全局 L1 非结构化剪枝
    parameters_to_prune = global_prune(
        model,
        amount=config.prune_amount
    )

    # 计算剪枝后的 Query 稀疏率
    sparsity = calculate_sparsity(
        parameters_to_prune
    )

    # 直接评估刚剪完的模型
    pruned_result = evaluate(
        model,
        test_loader,
        device
    )

    # 输出刚剪完的结果
    print(
        f"刚剪枝：{pruned_result}, "
        f"Query Sparsity={sparsity:.4f}"
    )

    # 进行 Recovery Fine-tuning
    model = recovery_finetune(
        model,
        train_loader,
        dev_loader,
        parameters_to_prune
    )

    # 在正式 remove 之前再次计算稀疏率
    final_sparsity = calculate_sparsity(
        parameters_to_prune
    )

    # 把 mask 正式融合回 weight
    remove_pruning(
        parameters_to_prune
    )

    # 恢复训练后再在 Test 集评估一次
    final_result = evaluate(
        model,
        test_loader,
        device
    )

    # 输出最终结果
    print(
        f"恢复训练后：{final_result}, "
        f"Query Sparsity={final_sparsity:.4f}"
    )

    # 保存 Transformers 模型
    model.save_pretrained(
        config.pruned_model_path
    )

    # 保存 Tokenizer
    tokenizer.save_pretrained(
        config.pruned_model_path
    )

    # 输出保存路径
    print(
        "剪枝模型已保存到：",
        config.pruned_model_path
    )


# 直接运行当前文件时执行 main
if __name__ == "__main__":
    main()
