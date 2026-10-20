import torch
from torch.utils.data import DataLoader, Dataset
from config import conf
from transformers import AutoTokenizer

def load_class_list():
    with open(conf.class_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def load_news_data(file_path, max_samples=None):
    if max_samples is not None and max_samples <= 0:
        raise ValueError('max_samples 必须大于 0')
    sample = []
    with open(file_path,'r',encoding='utf-8') as file:
        for idx,line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            pars = line.split('\t',1)
            if len(pars) != 2:
                raise ValueError(f"Invalid line format: {idx+1}")
            text, label = pars
            sample.append((text, int(label)))
            if max_samples is not None and len(sample) >= max_samples:
                break
    return sample

# 自定义数据集类DataSet
class NewDataSet(Dataset):
    def __init__(self,samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        text, label = self.samples[idx]
        return text,label

# 批处理函数
def build_context_fn(tokenizer, max_length=None):
    if max_length is None:
        max_length = conf.max_length
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        encoded['labels'] = torch.tensor(labels,dtype = torch.long)
        return encoded
    return collate_fn



def build_dataloader(file_path,tokenizer,shuffle, max_samples=None, config=None):
    config = conf if config is None else config
    samples=load_news_data(file_path, max_samples=max_samples)
    if not samples:
        raise ValueError(f'数据文件没有有效样本：{file_path}')
    dataset=NewDataSet(samples)
    loader=DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=build_context_fn(tokenizer, max_length=config.max_length),
        num_workers=config.num_workers,
        pin_memory=config.device.type == 'cuda',
    )
    return loader


if __name__ == '__main__':
    print(load_class_list()[:5])
    print(load_class_list())
    file_path = conf.train_path
    tokenizer = AutoTokenizer.from_pretrained(conf.pretrained_model_path)
    loader = build_dataloader(file_path,tokenizer,True)
    print(loader)
