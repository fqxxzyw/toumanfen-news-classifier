# 数据处理流程：读取文本文件 -> 保存为 Dataset -> 由 DataLoader 分批取样
#               -> collate_fn 分词、补齐并生成模型输入张量。
import torch
# Dataset 描述“有哪些样本以及如何取样”，DataLoader 负责批量取样与迭代。
from torch.utils.data import DataLoader, Dataset
# conf 是 config.py 中已经创建好的默认配置对象。
from config import conf
# 下方直接运行本文件的示例会用到分词器，训练时则由 train.py 传入分词器。
from transformers import AutoTokenizer

# 读取类别名称，列表下标就是类别 ID，顺序必须与数据中的数字标签保持一致。
# 例如 ['财经', '体育'] 对应 0=财经、1=体育；类别文件应每行一个名称且无空行。
def load_class_list():
    with open(conf.class_path, 'r', encoding='utf-8') as file:
        # strip() 去除每行首尾的空格、换行符，返回普通字符串列表。
        return [line.strip() for line in file]

# 从文件读取新闻，返回 [(文本, 标签ID), ...]。
# file_path 必须是路径字符串或 Path 对象，不能直接传整个 Config 对象。
# 文件每条有效行的格式为：新闻文本\t整数标签，例如“球队夺冠\t7”。
# max_samples=None 表示读取全部有效样本；传正整数可限制数量，用于快速调试。
def load_news_data(file_path, max_samples=None):
    # 避免传入 0 或负数，导致样本数量限制的含义不明确。
    if max_samples is not None and max_samples <= 0:
        raise ValueError('max_samples 必须大于 0')
    # 保存所有读取到的有效样本。
    sample = []
    with open(file_path,'r',encoding='utf-8') as file:
        # enumerate 从 0 开始计数，因此报错时用 idx + 1 显示实际文件行号。
        for idx,line in enumerate(file):
            line = line.strip()
            # 空白行不作为样本，也不计入 max_samples 限制的有效样本数量。
            if not line:
                continue
            # 最多按第一个制表符拆成两部分：新闻文本与标签。
            # 此格式要求新闻文本本身不包含制表符，否则无法按预期解析标签。
            pars = line.split('\t',1)
            if len(pars) != 2:
                raise ValueError(f"Invalid line format: {idx+1}")
            # 缺少分隔符的行在上面直接报错，避免把格式错误的数据静默跳过。
            text, label = pars
            # 文件读出的 label 是字符串，转成整数后才能作为分类标签。
            sample.append((text, int(label)))
            # 达到样本数量上限后立即停止读取，避免调试时加载整个大文件。
            if max_samples is not None and len(sample) >= max_samples:
                break
    return sample

# 自定义 Dataset：这里只保存原始文本和标签，暂时不分词、不移动到 GPU。
# 继承 Dataset 后，实现长度和下标取样方法即可配合 DataLoader 使用。
class NewDataSet(Dataset):
    def __init__(self,samples):
        # samples 是 load_news_data 返回的列表。
        self.samples = samples
    # len(dataset) 会调用此方法，返回样本总数。
    def __len__(self):
        return len(self.samples)
    # dataset[idx] 会调用此方法，返回一条 (新闻文本, 标签ID)。
    def __getitem__(self,idx):
        text, label = self.samples[idx]
        return text,label

# 创建一个批处理函数，而不是在此处立即处理某批数据。
# 返回的 collate_fn 会记住这里传入的 tokenizer 和 max_length（闭包）。
# DataLoader 每次取出一批样本后，会自动调用这个函数进行整理。
def build_context_fn(tokenizer, max_length=None):
    # 未显式指定截断长度时，采用默认配置中的值。
    if max_length is None:
        max_length = conf.max_length
    # batch 类似 [('球队夺冠', 7), ('股票上涨', 2)]。
    def collate_fn(batch):
        # 将“文本和标签配对的列表”拆成文本列表与标签列表。
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # 对整批文本一起分词，编码结果是包含多个输入张量的 BatchEncoding。
        # 张量通常包括 input_ids、attention_mask，某些模型还需要 token_type_ids。
        encoded = tokenizer(
            texts,
            # 动态补齐到当前批次中最长的编码长度，而不是一律补到 max_length。
            padding=True,
            # 先把过长文本截断，避免超过长度限制。
            truncation=True,
            max_length=max_length,
            # 返回 PyTorch 张量，输入张量形状为 [批次大小, 补齐后的词元长度]。
            return_tensors='pt'
        )
        # 标签形状为 [批次大小]；单标签分类需要整数类别 ID，使用 torch.long。
        # 键名使用 labels，模型收到它后就能自动计算分类损失。
        encoded['labels'] = torch.tensor(labels,dtype = torch.long)
        return encoded
    # 返回函数对象供 DataLoader 后续调用，此处不能写成 collate_fn()。
    return collate_fn



# 构建可迭代的批次加载器，训练、验证和测试共用此入口。
# 调用示例：build_dataloader(config.test_path, tokenizer, shuffle=False, config=config)。
# 注意第三个参数是 shuffle，不是 batch_size；批大小从 config.batch_size 读取。
def build_dataloader(file_path,tokenizer,shuffle, max_samples=None, config=None):
    # 如果调用方传入了配置就使用它，否则回退到默认配置 conf。
    config = conf if config is None else config
    samples=load_news_data(file_path, max_samples=max_samples)
    # 没有有效样本时提前给出清晰提示，避免后续训练中出现空数据或除零问题。
    if not samples:
        raise ValueError(f'数据文件没有有效样本：{file_path}')
    # 将普通样本列表包装成 Dataset，再交给 DataLoader 分批加载。
    dataset=NewDataSet(samples)
    loader=DataLoader(
        dataset,
        # 每批样本数；最后一批可以小于该值，因为默认不丢弃不足一批的样本。
        batch_size=config.batch_size,
        # 训练通常为 True，打乱样本顺序；验证和测试通常为 False。
        shuffle=shuffle,
        # 创建并传入批处理函数，由 DataLoader 在每批取样后调用。
        collate_fn=build_context_fn(tokenizer, max_length=config.max_length),
        # 数据加载子进程数量；0 表示在当前进程加载，便于 Windows 环境调试。
        num_workers=config.num_workers,
        # 使用 CUDA 时启用锁页内存，便于后续将 CPU 张量传输到 GPU。
        # 此选项本身不会把数据移动到 GPU，实际移动由训练/评估代码负责。
        pin_memory=config.device.type == 'cuda',
    )
    # 遍历 loader 时，每次得到一个已分词、含 labels 的批次字典。
    return loader


# 直接运行本文件时执行读取示例；其他模块导入本文件时不会执行这个示例。
if __name__ == '__main__':
    print(load_class_list()[:5])
    print(load_class_list())
    file_path = conf.train_path
    tokenizer = AutoTokenizer.from_pretrained(conf.pretrained_model_path)
    loader = build_dataloader(file_path,tokenizer,True)
    # 这里只打印 DataLoader 对象；实际查看数据可通过 next(iter(loader)) 获取一批。
    print(loader)
