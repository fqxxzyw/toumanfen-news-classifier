# 导包
import json

import numpy as np
import torch
from tqdm import tqdm
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from data import load_class_list, build_dataloader
from metrics import cal_metrics
import random

config = Config()


# 固定随机种子
def set_seed(seed):
    random.seed(seed)  # python 随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # GPU的随机种子

# 将一个batch移动到指定设备
def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

# 模型的评估
def evaluate(model, data_loader, device,class_name):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = move_batch_to_device(batch, device)
            # 前向传播，得到预测结果
            outputs = model(**batch)
            # 拿到预测结果
            logits = outputs.logits
            # 获取分数最大预测结果
            pred = torch.argmax(logits,dim = 1)
            y_true.extend(batch['labels'].cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return cal_metrics(y_true,y_pred,class_name)





# 训练函数

def train():
    # 1.固定随机种子
    set_seed(config.seed)
    # 2.数据准备
    #  加载模型
    class_list = load_class_list()
    num_classes = len(class_list)
    #  类别->ID
    class_to_id = {class_name : i for i,class_name in enumerate(class_list)}
    id_to_class = {i: class_name for i ,class_name in enumerate(class_list)}
    # 3.加载模型bert模型
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=num_classes,
        id2label = id_to_class,
        label2id = class_to_id
    )
    model.to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
    # 4.构建dataloader
    train_dataloader = build_dataloader(config.train_path, tokenizer, shuffle=True, config=config)
    dev_dataloader = build_dataloader(config.dev_path, tokenizer, shuffle=False, config=config)
    # 5.优化器，学习率，损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    total_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    best_f1 = float('-inf')
    no_improvement_count = 0

    # 6.循环训练（核心步骤）
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader,desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress_bar:
            batch = move_batch_to_device(batch, config.device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
            optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        train_avg_loss = total_loss / len(train_dataloader)
        dev_metrics = evaluate(model, dev_dataloader, config.device,class_list)
        dev_f1 = dev_metrics[3]
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_avg_loss:.4f}, Dev F1: {dev_f1:.4f}")
        # 7.根据评估中的F1找到最优的模型并保存
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            no_improvement_count = 0
            model.save_pretrained(config.best_model_dir)
            tokenizer.save_pretrained(config.best_model_dir)
            metadata = {
                'class_list' : class_list,
                'max_length' : config.max_length,
                'best_f1' : best_f1,
                'pretrained_mode':str(config.pretrained_model_path)
            }
            with open(config.best_model_dir / 'metadata.json','w',encoding='utf-8') as f:
                json.dump(metadata,f,ensure_ascii=False,indent=2)
            print('最优模型已经保存')
        else:
            # 8.早停
            no_improvement_count += 1
            if no_improvement_count >= config.patience:
                print(f"Early stopping after {epoch + 1} epochs, best F1: {best_f1:.4f}")
                break




if __name__ == '__main__':
    train()



