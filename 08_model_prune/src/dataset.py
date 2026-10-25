# 导入 Path
from pathlib import Path
# 导入类型提示
from typing import Callable
# 导入 PyTorch
import torch
# 导入 Dataset 和 DataLoader
from torch.utils.data import Dataset, DataLoader


def _parse_line(
    line: str
):
    # 去掉行首和行尾空白字符
    line = line.strip()

    # 空行直接忽略
    if not line:
        return None

    # 支持格式 1：0<TAB>这是一条新闻
    if "\t" in line:
        # 只按第一个 Tab 切分
        first, second = line.split(
            "\t",
            1
        )

        # 如果第一列是数字，就认为第一列是标签
        if first.strip().isdigit():
            return (
                second.strip(),
                int(first.strip())
            )

        # 如果第二列是数字，也兼容“文本<TAB>标签”
        if second.strip().isdigit():
            return (
                first.strip(),
                int(second.strip())
            )

    # 支持格式 2：__label__3 这是一条新闻
    if line.startswith("__label__"):
        # 按第一个空格切开标签和文本
        label_part, text = line.split(
            maxsplit=1
        )

        # 删除 __label__ 前缀
        label = label_part.replace(
            "__label__",
            ""
        )

        # 返回文本和整数标签
        return (
            text.strip(),
            int(label)
        )

    # 如果格式不符合要求，抛出清晰错误
    raise ValueError(
        "数据格式错误。推荐每行使用：标签<TAB>文本，例如：0\\t这是一条体育新闻"
    )


def load_samples(
    file_path
):
    # 转成 Path 对象
    file_path = Path(
        file_path
    )

    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{file_path}"
        )

    # 创建样本列表
    samples = []

    # 以 UTF-8 编码打开文件
    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        # 一行一行读取
        for line_number, line in enumerate(
            file,
            start=1
        ):
            # 跳过注释行
            if line.lstrip().startswith("#"):
                continue

            # 解析当前行
            try:
                item = _parse_line(
                    line
                )
            except Exception as exc:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行解析失败：{line.strip()}"
                ) from exc

            # 空行不加入数据集
            if item is not None:
                samples.append(
                    item
                )

    # 返回所有样本
    return samples


def load_classes(
    file_path
):
    # 转成 Path
    file_path = Path(
        file_path
    )

    # 类别文件不存在时返回空列表
    if not file_path.exists():
        return []

    # 打开类别文件
    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        # 每一行代表一个类别名称
        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    # 返回类别名称列表
    return classes


class NewsDataset(
    Dataset
):
    def __init__(
        self,
        samples
    ):
        # 保存全部样本
        self.samples = samples

    def __len__(
        self
    ):
        # 返回样本数量
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):
        # 读取一条文本和标签
        text, label = self.samples[
            index
        ]

        # 返回文本、标签和原始下标
        return {
            "text": text,
            "label": label,
            "index": index
        }


def build_collate_fn(
    tokenizer,
    max_length
) -> Callable:
    # 返回真正给 DataLoader 使用的批处理函数
    def collate_fn(
        batch
    ):
        # 取出一个 Batch 中的所有文本
        texts = [
            item["text"]
            for item in batch
        ]

        # 取出所有标签
        labels = torch.tensor(
            [
                item["label"]
                for item in batch
            ],
            dtype=torch.long
        )

        # 取出原始样本下标，后面读取 Teacher 缓存时会使用
        indices = torch.tensor(
            [
                item["index"]
                for item in batch
            ],
            dtype=torch.long
        )

        # 使用 Teacher 的 Tokenizer 批量编码
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 返回训练需要的字段
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "indices": indices,
            "texts": texts
        }

    # 返回批处理函数
    return collate_fn


def build_dataloader(
    file_path,
    tokenizer,
    max_length,
    batch_size,
    shuffle,
    num_workers=0
):
    # 加载原始样本
    samples = load_samples(
        file_path
    )

    # 创建 Dataset
    dataset = NewsDataset(
        samples
    )

    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collate_fn(
            tokenizer,
            max_length
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # 返回 DataLoader
    return loader