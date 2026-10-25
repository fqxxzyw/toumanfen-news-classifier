import torch
from tqdm import tqdm
from config import config,device
from dataset import build_dataloader
from teacher import load_teacher


# 编写保存老师模型预测的结果函数
def save_teacher_cache():
    # 加载老师模型
    tokenizer,teacher = load_teacher(config.teacher_path,config.num_classes)
    # 将teacher模型转换为cuda设备
    teacher.to(device)
    # 创建训练集
    train_loader = build_dataloader(
        str(config.train_path),
        tokenizer,
        config.max_length,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # 保存所有老师的Logits
    all_logits = []
    # 保存所有老师的hidden
    all_hidden=[]
    # 保存样本下标
    all_indices=[]
    teacher.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader,desc='Cache teacher'):
            # input_ids
            input_ids = batch['input_ids'].to(device)
            # mask
            mask = batch['attention_mask'].to(device)

            # teacher
            outputs = teacher(
                input_ids=input_ids,
                attention_mask = mask,
                output_hidden_states = True,
                return_dict = True
            )
            teacher_logits = outputs.logits.detach().cpu()
            teacher_hidden = outputs.hidden_states[-1][:,0,:].detach().cpu()

            all_logits.append(teacher_logits)

            all_hidden.append(teacher_hidden)

            all_indices.append(batch['indices'].cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_hidden = torch.cat(all_hidden, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    # 按照原始数据重排序
    order = torch.argsort(all_indices, dim=0)
    all_logits = all_logits[order]
    all_hidden = all_hidden[order]
    all_indices = all_indices[order]

    # 保存老师模型预测的结果
    torch.save(all_logits, config.teacher_logits_cache)
    torch.save(all_hidden, config.teacher_hidden_cache)
    torch.save(all_indices, config.teacher_indices_cache)
    print('Teacher cache saved successfully!')
    print(f"老师模型的logits的形状:{all_logits.shape}")
    print(f"老师模型的hidden的形状:{all_hidden.shape}")
    print(f"老师模型的indices的形状:{all_indices.shape}")
if __name__ == '__main__':
    save_teacher_cache()
