"""读取新闻测试集，按批次分词，为 FP32 与 INT8 评估提供相同的输入。"""

import torch
from torch.utils.data import DataLoader, Dataset

from config import config as default_config


def load_data(file_path, max_samples=None):
    """读取“新闻标题<Tab>整数标签”；max_samples 为有效样本数量上限。"""
    if max_samples is not None and (not isinstance(max_samples, int) or isinstance(max_samples, bool)
                                    or max_samples <= 0):
        raise ValueError('max_samples 必须为正整数或 None。')
    samples = []
    # utf-8-sig 兼容普通 UTF-8 和带 BOM 的 UTF-8 文件。
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip('\r\n')
            if not line.strip():
                continue  # 空白行不占用 max_samples 的有效样本名额。
            # 从最后一个 Tab 拆分，标题本身带 Tab 时仍能正确提取末尾标签。
            parts = line.rsplit('\t', 1)
            if len(parts) != 2 or not parts[0].strip():
                raise ValueError(f'第 {line_number} 行格式错误，应为非空标题、Tab、整数标签。')
            text, label_text = parts
            try:
                label = int(label_text.strip())
            except ValueError as error:
                raise ValueError(f'第 {line_number} 行的标签不是整数。') from error
            if label < 0:
                raise ValueError(f'第 {line_number} 行的标签不能为负数。')
            samples.append((text.strip(), label))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


class NewDataSet(Dataset):
    """保存原始文本与标签；实际分词推迟到每批数据组装时执行。"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # 返回一条 (text, label_id)。


def build_context_fn(tokenizer, max_length=None):
    """创建 DataLoader 的 collate_fn，保留原来的函数名以兼容调用。"""
    max_length = default_config.max_length if max_length is None else max_length
    if max_length <= 0:
        raise ValueError('max_length 必须大于 0。')

    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # 动态补齐到本批最长序列；超长标题截断，避免超过模型长度限制。
        # 编码张量形状为 [batch_size, sequence_length]，此时仍在 CPU 上。
        encoded = tokenizer(texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors='pt')
        # 分类标签是一维整数张量，不能使用浮点数类型。
        encoded['labels'] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate_fn  # 返回函数本身，由 DataLoader 在每个批次调用。


def build_dataloader(file_path, tokenizer, shuffle=False, max_samples=None, config=None):
    """构建可重复遍历的 DataLoader；评估默认不打乱，也不丢弃最后的小批次。"""
    config = default_config if config is None else config
    samples = load_data(file_path, max_samples=max_samples)
    if not samples:
        raise ValueError(f'数据文件没有有效样本：{file_path}')
    if config.batch_size <= 0:
        raise ValueError('batch_size 必须大于 0。')

    # torch.device 同时接受 'cpu' 字符串与 torch.device 对象。
    device = torch.device(config.device)
    return DataLoader(
        NewDataSet(samples),
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=build_context_fn(tokenizer, max_length=config.max_length),
        # 当前 Config 没有 num_workers 时使用 0，在 Windows 当前进程内加载。
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=device.type == 'cuda',
    )
