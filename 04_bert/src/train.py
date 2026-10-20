# BERT 新闻分类训练入口。
# 整体流程：读取类别 -> 加载预训练模型和分词器 -> 构建批次 -> 循环训练
#           -> 每轮验证 -> 保存最优模型 -> 连续多轮无提升时提前结束。
# batch 是一次送入模型的一小批样本；epoch 是完整遍历一遍训练集。

# json 用于保存类别名称、最大输入长度和最优 F1 等辅助信息。
import json

# numpy、torch 和 random 分别提供数组计算、深度学习及 Python 随机数功能。
import numpy as np
import torch
# tqdm 为训练和评估添加进度条。
from tqdm import tqdm
# Config 集中管理数据路径、模型路径、训练参数和 CPU/GPU 设备。
from config import Config
# AutoTokenizer：把文本转成模型输入的数字 ID。
# AutoModelForSequenceClassification：根据配置加载带文本分类头的模型。
# get_linear_schedule_with_warmup：创建先预热、再线性下降的学习率调度器。
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
# data.py 负责读取类别列表，以及把数据集组织成可按批次遍历的 DataLoader。
from data import load_class_list, build_dataloader
# metrics.py 负责计算准确率、宏平均精确率、召回率、F1 和分类报告。
from metrics import cal_metrics
import random

# 创建配置对象，后面的函数通过 config.xxx 读取配置。
config = Config()


# 固定各随机数生成器的种子，减少多次实验之间的随机差异。
# 相同种子有助于复现实验，但不保证不同硬件和所有 GPU 算法逐位一致。
def set_seed(seed):
    random.seed(seed)  # python 随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # GPU的随机种子

# 将一个 batch 中的所有张量移到同一设备，返回新的字典。
# batch 通常包含 input_ids（词元 ID）、attention_mask（有效位置标记）、
# token_type_ids（如果分词器提供）和 labels（真实类别 ID）。
# 字典推导式中的 k 是字段名，v 是对应的张量；v.to(device) 负责移动设备。
# 模型参数和输入张量必须在同一设备上，例如都位于 CPU 或都位于同一块 GPU。
def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

# 在验证集或测试集上评估模型，只预测、统计指标，不更新模型参数。
# class_name 是按类别 ID 顺序排列的类别名称列表。
# 返回值沿用 cal_metrics 的元组：(acc, precision, recall, f1, report)。
def evaluate(model, data_loader, device,class_name):
    # 切换到评估模式，让 Dropout 等层采用评估时的行为。
    # eval() 本身不会关闭梯度记录，因此下面还需要 torch.no_grad()。
    model.eval()
    # 收集整个数据集的真实标签与预测标签，最后统一计算指标。
    # 不直接平均各 batch 的 F1，因为这通常不等于整个数据集的 F1。
    y_true = []
    y_pred = []

    # 评估时不需要反向传播；关闭梯度记录可减少内存占用。
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = move_batch_to_device(batch, device)
            # **batch 把字典展开为关键字参数，相当于传入
            # model(input_ids=..., attention_mask=..., labels=..., ...)。
            outputs = model(**batch)
            # logits 是每个类别的原始分数，形状为 [当前批次样本数, 类别数]。
            # 它还不是概率；只判断最大分数所在类别时，不必先做 softmax。
            logits = outputs.logits
            # dim=1 表示沿类别维度取最大值的下标，得到每条样本的预测类别 ID。
            pred = torch.argmax(logits,dim = 1)
            # sklearn 指标函数使用 CPU 上的数据，因此先 cpu() 再转 numpy。
            # extend 将这批标签逐个加入列表，而不是把整个数组当作一个元素。
            y_true.extend(batch['labels'].cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # 计算并打印整个评估集的指标，供 train() 选择最优模型。
    return cal_metrics(y_true,y_pred,class_name)





# 完整训练流程。函数定义时不会立即训练，文件末尾的入口负责调用它。

def train():
    # 1. 在加载模型和创建 DataLoader 之前固定随机种子。
    # 这样可控制分类头随机初始化、训练样本打乱等操作中的随机性。
    set_seed(config.seed)
    # 2. 读取类别列表，列表的顺序必须与数据文件中的数字标签保持一致。
    # 例如 ['财经', '体育', '科技'] 对应 0、1、2，共有 3 个类别。
    class_list = load_class_list()
    # 类别数量决定分类头每条样本输出多少个分数；先赋值，后面才能引用。
    num_classes = len(class_list)
    # 构建双向映射，方便在类别名称与数字 ID 之间转换。
    # class_to_id：例如 {'财经': 0, '体育': 1}。
    # id_to_class：例如 {0: '财经', 1: '体育'}。
    class_to_id = {class_name : i for i,class_name in enumerate(class_list)}
    id_to_class = {i: class_name for i ,class_name in enumerate(class_list)}
    # 3. 从本地预训练目录加载 BERT，并配置用于当前任务的分类头。
    # num_labels 指定类别数；id2label 和 label2id 会写入模型配置，保存后可复用。
    # 当原始目录只有预训练 BERT 时，分类头可能随机初始化，需要接下来的训练。
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=num_classes,
        id2label = id_to_class,
        label2id = class_to_id
    )
    # 将整个模型移到配置指定的设备；之后每个 batch 也需要移到这里。
    model.to(config.device)
    # 使用与预训练模型配套的分词器，保证词元 ID 与模型词表一致。
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
    # 4. 构建 DataLoader：读取文本和标签，按 batch_size 分批并完成分词。
    # 训练集 shuffle=True，每轮打乱样本顺序；验证集无需打乱。
    # 显式传入 config，供 data.py 读取批大小、最大长度和加载进程数等配置。
    train_dataloader = build_dataloader(config.train_path, tokenizer, shuffle=True, config=config)
    dev_dataloader = build_dataloader(config.dev_path, tokenizer, shuffle=False, config=config)
    # 5. 创建优化器和学习率调度器。
    # AdamW 根据梯度更新参数；lr 是学习率，weight_decay 是权重衰减系数。
    # model.parameters() 包括 BERT 和分类头的参数，此处会一起进行微调。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    # len(train_dataloader) 是每轮的批次数，每处理一批就更新一次参数。
    # total_steps 是训练计划的总更新次数；若触发早停，实际次数会更少。
    total_steps = len(train_dataloader) * config.num_epochs
    # 预热步数按总步数的一定比例计算，int 表示向下取整。
    warmup_steps = int(total_steps * config.warmup_ratio)
    # 预热阶段逐渐增加学习率，随后逐渐降低至 0。
    # 调度器绑定 optimizer，后面每次参数更新后调用一次 scheduler.step()。
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # 这两个变量必须在 epoch 循环外初始化，才能保留各轮之间的历史状态。
    # 负无穷小于任何正常 F1，保证第一次得到有限 F1 时可以保存模型。
    best_f1 = float('-inf')
    # 连续未提升的轮数：有提升时清零，没有提升时加 1。
    no_improvement_count = 0

    # 6. 循环训练，每个 epoch 完整遍历一次训练集。
    for epoch in range(config.num_epochs):
        # evaluate() 会切到评估模式，因此每轮训练开始都要重新切回训练模式。
        # model.train() 控制 Dropout 等层的行为，本身不会执行参数更新。
        model.train()
        # 只统计当前这一轮的损失，因此每轮开始时清零。
        total_loss = 0.0
        # epoch 从 0 开始计数；显示时加 1，让第一轮显示为 Epoch 1。
        progress_bar = tqdm(train_dataloader,desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress_bar:
            # 6.1 将当前批次的输入和真实标签移到模型所在的设备。
            batch = move_batch_to_device(batch, config.device)
            # 6.2 清除上一批的梯度。PyTorch 默认会累加梯度，因此需要显式清零。
            optimizer.zero_grad()
            # 6.3 前向传播：输入经过 BERT 和分类头，得到分类分数和损失。
            # batch 含 labels，模型会自动计算当前单标签多分类任务的交叉熵损失。
            outputs = model(**batch)
            loss = outputs.loss
            # 6.4 反向传播：计算损失对各个可训练参数的梯度，此时尚未更新参数。
            loss.backward()
            # 6.5 梯度裁剪：限制参数梯度的整体范数，缓解梯度过大的问题。
            # 应在 backward() 之后、optimizer.step() 之前执行。
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
            # 6.6 优化器根据梯度和当前学习率更新模型参数。
            optimizer.step()

            # 6.7 更新学习率，准备下一批训练使用。
            scheduler.step()
            # item() 将标量张量转为 Python 数值，便于累计而不保留计算图。
            total_loss += loss.item()

        # 当前写法对各批次的平均损失再求平均，用于观察训练趋势。
        # 如果最后一批大小不同，它不严格等于按每条样本加权的平均损失。
        train_avg_loss = total_loss / len(train_dataloader)
        # 每轮训练结束后评估一次验证集，衡量模型对未参与梯度更新的数据的表现。
        dev_metrics = evaluate(model, dev_dataloader, config.device,class_list)
        # cal_metrics 返回 (acc, precision, recall, f1, report)，所以下标 3 是 F1。
        dev_f1 = dev_metrics[3]
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_avg_loss:.4f}, Dev F1: {dev_f1:.4f}")
        # 7. 使用验证集宏平均 F1 选择最优模型。
        # 必须严格大于历史最好成绩才算提升；相等也会进入下面的早停计数。
        if dev_f1 > best_f1:
            # 更新历史最好成绩，并重置“连续未提升”的轮数。
            best_f1 = dev_f1
            no_improvement_count = 0
            # 保存完整的微调模型权重和模型配置，而不只是分类头。
            # 再保存配套分词器，之后可直接从 best_model 目录加载并预测。
            # 每次提升都会向同一目录写入当前最优版本，不会为每轮另存一份。
            model.save_pretrained(config.best_model_dir)
            tokenizer.save_pretrained(config.best_model_dir)
            # 额外保存预测时有用的信息：类别顺序、截断长度及本次最优分数。
            # pretrained_mode 是现有元数据字段名，用于记录训练来源路径；
            # 加载完整 best_model 做预测时，不要求同桌电脑存在这个原始路径。
            metadata = {
                'class_list' : class_list,
                'max_length' : config.max_length,
                'best_f1' : best_f1,
                # Path 对象不能直接写入 JSON，因此先转换成字符串。
                'pretrained_mode':str(config.pretrained_model_path)
            }
            # / 用于拼接 Path 路径；'w' 表示覆盖写入，UTF-8 用于支持中文。
            # with 会在写入完成后自动关闭文件。
            with open(config.best_model_dir / 'metadata.json','w',encoding='utf-8') as f:
                # ensure_ascii=False 保留中文；indent=2 让 JSON 按两空格缩进。
                json.dump(metadata,f,ensure_ascii=False,indent=2)
            print('最优模型已经保存')
        else:
            # 8. 早停：当前轮没有提升，连续未提升次数加 1。
            no_improvement_count += 1
            # patience 来自 Config，例如设为 2，表示连续 2 轮无提升就停止。
            # 等待次数按 epoch 计算，而不是按 batch 计算。
            if no_improvement_count >= config.patience:
                print(f"Early stopping after {epoch + 1} epochs, best F1: {best_f1:.4f}")
                # 跳出 epoch 循环，结束训练；磁盘上仍保存之前 F1 最好的模型。
                # 注意：这里没有把内存中的模型重新加载成最优版本。
                break




# 直接运行本文件时开始训练；其他文件 import train 时不会自动调用 train()。
if __name__ == '__main__':
    train()


