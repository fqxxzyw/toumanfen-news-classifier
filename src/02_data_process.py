"""步骤 02：清洗文本、分词并导出 CSV。"""
import os                                                   # 导入os模块，用于处理文件路径
import re                                                   # 导入正则表达式模块，用于清洗文本
from collections import Counter                            # 导入Counter，用于统计类别数量

import jieba                                                # 导入jieba，用于中文分词
import numpy as np                                          # 导入numpy，用于计算统计指标
import pandas as pd                                         # 导入pandas，用于处理表格数据


DATA_DIR = os.path.join(                                   # 根据脚本位置定位数据目录
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
TRAIN_INPUT_PATH = os.path.join(DATA_DIR, "train.txt")      # 定义原始训练集路径
DEV_INPUT_PATH = os.path.join(DATA_DIR, "dev.txt")          # 定义原始验证集路径
TEST_INPUT_PATH = os.path.join(DATA_DIR, "test.txt")        # 定义原始测试集路径

TRAIN_OUTPUT_PATH = os.path.join(DATA_DIR, "train.csv")     # 定义处理后训练集路径
DEV_OUTPUT_PATH = os.path.join(DATA_DIR, "dev.csv")         # 定义处理后验证集路径
TEST_OUTPUT_PATH = os.path.join(DATA_DIR, "test.csv")       # 定义处理后测试集路径

MAX_WORDS = 30                                              # 设置words字段最多保留30个词
NUM_CLASSES = 10                                            # 设置新闻类别数量
DROP_DUPLICATES = False                                     # 是否删除数据集内部的完全重复样本


def read_txt_dataset(file_path):
    """
    读取原始TXT新闻分类数据。

    参数：
        file_path：原始TXT文件路径。

    返回：
        包含text和label字段的DataFrame。
    """

    if not os.path.exists(file_path):                       # 判断原始文件是否存在
        raise FileNotFoundError(                            # 文件不存在时抛出异常
            f"数据文件不存在：{file_path}"                  # 输出不存在的文件路径
        )

    data = pd.read_csv(                                     # 使用pandas读取TXT文件
        file_path,                                          # 指定原始TXT文件路径
        sep="\t",                                           # 指定字段使用制表符分隔
        encoding="utf-8"                                    # 指定文件编码为UTF-8
    )

    if {"sentence", "label"}.issubset(data.columns):        # 判断原始文件是否包含sentence和label
        data = data.rename(                                 # 重命名原始字段
            columns={
                "sentence": "text"                          # 将sentence统一改为text
            }
        )

    elif {"text", "label"}.issubset(data.columns):          # 判断文件是否已经使用text和label字段
        data = data.copy()                                  # 创建数据副本

    else:                                                   # 如果没有识别到正常表头
        data = pd.read_csv(                                 # 按照无表头格式重新读取
            file_path,                                      # 指定原始TXT文件路径
            sep="\t",                                       # 指定字段使用制表符分隔
            names=["text", "label"],                        # 为两列指定字段名称
            header=None,                                    # 表示原始文件没有表头
            encoding="utf-8"                                # 指定文件编码为UTF-8
        )

    if data.shape[1] != 2:                                  # 检查原始数据是否正好为两列
        raise ValueError(                                   # 字段数量异常时抛出错误
            f"{file_path}读取后不是两列，请检查分隔符"       # 提示检查TXT分隔格式
        )

    data = data[["text", "label"]]                          # 只保留文本和标签两个原始字段

    print(f"{file_path}读取成功")                           # 输出文件读取成功信息
    print(f"原始样本数量：{len(data)}")                     # 输出原始样本数量
    print(data.head(10))                                    # 输出前10条数据

    return data                                             # 返回读取后的DataFrame


def show_basic_statistics(data, dataset_name):
    """
    输出数据处理前的基本统计信息。

    参数：
        data：原始数据集。
        dataset_name：数据集名称。
    """

    label_count = Counter(                                  # 统计每个类别的样本数量
        data["label"].dropna().tolist()                     # 删除空标签后转换为列表
    )

    total = len(data)                                       # 统计数据集总样本数量

    print(f"\n{dataset_name}样本总量：{total}")              # 输出样本总量
    print(f"{dataset_name}类别数量：{len(label_count)}")     # 输出类别数量

    for label, number in sorted(label_count.items()):       # 按标签编号遍历统计结果
        ratio = number / total * 100                        # 计算当前类别的样本比例

        print(                                              # 输出类别统计信息
            f"标签{label}："
            f"样本数={number}，"
            f"占比={ratio:.2f}%"
        )

    text_length = data["text"].fillna("").astype(str).apply(  # 获取原始文本长度
        len                                                   # 计算每条标题的字符数量
    )

    length_mean = np.mean(text_length)                      # 计算文本长度均值
    length_std = np.std(text_length)                        # 计算文本长度标准差

    print(f"{dataset_name}平均文本长度：{length_mean:.2f}")  # 输出平均文本长度
    print(f"{dataset_name}文本长度标准差：{length_std:.2f}")  # 输出文本长度标准差


def clean_text(text):
    """
    清洗单条新闻标题。

    参数：
        text：原始新闻标题。

    返回：
        清洗后的新闻标题。
    """

    text = str(text)                                        # 将输入统一转换为字符串

    text = re.sub(                                          # 删除HTML标签
        r"<[^>]+>",                                         # 匹配HTML标签
        " ",                                                # 使用空格替换
        text                                                # 指定待处理文本
    )

    text = re.sub(                                          # 处理换行符和制表符
        r"[\r\n\t]",                                        # 匹配换行符、回车符和制表符
        " ",                                                # 使用空格替换
        text                                                # 指定待处理文本
    )

    text = re.sub(                                          # 删除不可见字符
        r"[\u200b-\u200f\ufeff]",                           # 匹配零宽字符和BOM
        "",                                                 # 使用空字符串替换
        text                                                # 指定待处理文本
    )

    text = re.sub(                                          # 处理乱码替换符
        r"\ufffd",                                          # 匹配乱码字符
        " ",                                                # 使用空格替换
        text                                                # 指定待处理文本
    )

    text = re.sub(                                          # 合并连续空白字符
        r"\s+",                                             # 匹配一个或多个空白字符
        " ",                                                # 替换为一个空格
        text                                                # 指定待处理文本
    )

    text = text.strip()                                     # 删除文本首尾空格

    return text                                             # 返回清洗后的文本


def cut_sentence(text):
    """
    使用jieba对新闻标题进行中文分词。

    参数：
        text：清洗后的新闻标题。

    返回：
        使用空格连接的分词结果。
    """

    word_list = jieba.lcut(text)                            # 使用jieba进行中文分词

    word_list = [                                           # 创建清洗后的词语列表
        word.strip()                                        # 删除词语首尾空格
        for word in word_list                               # 遍历全部分词结果
        if word.strip()                                     # 只保留非空词语
    ]

    words = " ".join(word_list)                             # 使用空格连接分词结果

    return words                                            # 返回分词后的文本


def truncate_words(words, max_words=MAX_WORDS):
    """
    按照词语数量截断分词结果。

    参数：
        words：使用空格连接的分词文本。
        max_words：最多保留的词语数量。

    返回：
        截断后的分词文本。
    """

    word_list = words.split()                               # 将words拆分为词语列表

    word_list = word_list[:max_words]                       # 最多保留前max_words个词

    words = " ".join(word_list)                             # 重新连接截断后的词语

    return words                                            # 返回截断结果


def process_dataset(data, dataset_name):
    """
    对一个数据集执行完整处理。

    参数：
        data：包含text和label字段的原始数据集。
        dataset_name：数据集名称。

    返回：
        包含text、label和words字段的标准数据集。
    """

    data = data.copy()                                      # 创建数据副本，避免修改原始数据

    original_size = len(data)                               # 记录处理前的样本数量

    data = data.dropna(                                     # 删除关键字段为空的样本
        subset=["text", "label"]                            # 指定text和label为关键字段
    )

    data["text"] = data["text"].astype(str)                 # 将文本统一转换为字符串

    data["label"] = pd.to_numeric(                          # 尝试将标签转换为数值
        data["label"],                                      # 指定标签字段
        errors="coerce"                                     # 转换失败的标签设置为空值
    )

    invalid_number_count = data["label"].isna().sum()       # 统计非数字标签数量

    if invalid_number_count > 0:                            # 判断是否存在非数字标签
        print(                                              # 输出异常标签数量
            f"{dataset_name}非数字标签数量："
            f"{invalid_number_count}"
        )

    data = data.dropna(subset=["label"])                    # 删除无法转换为数字的标签

    data["label"] = data["label"].astype(int)               # 将有效标签转换为整数

    invalid_range_mask = ~data["label"].between(            # 判断标签是否位于合法范围
        0,                                                  # 设置最小标签为0
        NUM_CLASSES - 1                                     # 设置最大标签为9
    )

    invalid_range_count = invalid_range_mask.sum()          # 统计越界标签数量

    if invalid_range_count > 0:                             # 判断是否存在越界标签
        print(                                              # 输出越界标签数量
            f"{dataset_name}越界标签数量："
            f"{invalid_range_count}"
        )

        data = data[~invalid_range_mask]                    # 删除标签越界的样本

    data["text"] = data["text"].apply(                      # 清洗新闻标题
        clean_text                                          # 调用统一文本清洗函数
    )

    empty_text_mask = data["text"].str.len() == 0           # 判断清洗后是否为空文本
    empty_text_count = empty_text_mask.sum()                # 统计空文本数量

    if empty_text_count > 0:                                # 判断是否存在空文本
        print(                                              # 输出空文本数量
            f"{dataset_name}空文本数量："
            f"{empty_text_count}"
        )

        data = data[~empty_text_mask]                       # 删除清洗后为空的样本

    if DROP_DUPLICATES:                                     # 判断是否需要删除完全重复样本
        before_drop_size = len(data)                        # 记录去重前的样本数量

        data = data.drop_duplicates(                        # 删除完全重复样本
            subset=["text", "label"],                       # 文本和标签都相同才视为重复
            keep="first"                                    # 保留第一次出现的样本
        )

        duplicate_count = before_drop_size - len(data)      # 计算被删除的重复样本数量

        print(                                              # 输出重复样本处理结果
            f"{dataset_name}删除完全重复样本数量："
            f"{duplicate_count}"
        )

    data["words"] = data["text"].apply(                     # 根据text生成words字段
        cut_sentence                                        # 使用jieba进行分词
    )

    data["words"] = data["words"].apply(                    # 对words字段执行长度截断
        lambda words: truncate_words(                       # 调用按词截断函数
            words,                                          # 传入当前分词文本
            MAX_WORDS                                       # 传入最大词语数量
        )
    )

    empty_words_mask = data["words"].str.len() == 0         # 判断words字段是否为空
    empty_words_count = empty_words_mask.sum()              # 统计空words数量

    if empty_words_count > 0:                               # 判断是否存在空words
        print(                                              # 输出空words数量
            f"{dataset_name}空words数量："
            f"{empty_words_count}"
        )

        data = data[~empty_words_mask]                      # 删除words为空的样本

    data = data[                                            # 固定最终字段名称和顺序
        ["text", "label", "words"]                          # 只保留三个目标字段
    ]

    data = data.reset_index(drop=True)                      # 重建连续的数据行索引

    processed_size = len(data)                              # 记录处理后的样本数量
    removed_size = original_size - processed_size           # 计算被删除的样本数量

    print(f"\n{dataset_name}处理完成")                      # 输出处理完成提示
    print(f"处理前样本数量：{original_size}")               # 输出处理前样本数量
    print(f"处理后样本数量：{processed_size}")              # 输出处理后样本数量
    print(f"删除样本数量：{removed_size}")                  # 输出被删除的样本数量

    return data                                             # 返回处理后的标准数据集


def validate_dataset(data, dataset_name):
    """
    验证处理后的数据集。

    参数：
        data：处理后的标准数据集。
        dataset_name：数据集名称。
    """

    expected_columns = [                                   # 定义最终要求的字段
        "text",                                             # 清洗后的完整文本
        "label",                                            # 类别标签
        "words"                                             # 分词后的文本
    ]

    actual_columns = data.columns.tolist()                  # 获取实际字段名称和顺序

    if actual_columns != expected_columns:                  # 判断字段是否符合要求
        raise ValueError(                                   # 字段错误时抛出异常
            f"{dataset_name}字段错误：{actual_columns}"      # 输出实际字段
        )

    if data.empty:                                          # 判断处理后的数据集是否为空
        raise ValueError(f"{dataset_name}处理后为空")        # 数据为空时抛出异常

    if data["text"].isna().any():                           # 检查text字段是否存在缺失值
        raise ValueError(f"{dataset_name}存在空text")        # 存在空值时抛出异常

    if data["label"].isna().any():                          # 检查label字段是否存在缺失值
        raise ValueError(f"{dataset_name}存在空label")       # 存在空值时抛出异常

    if data["words"].isna().any():                          # 检查words字段是否存在缺失值
        raise ValueError(f"{dataset_name}存在空words")       # 存在空值时抛出异常

    if not data["label"].between(0, NUM_CLASSES - 1).all():  # 检查标签范围是否正确
        raise ValueError(f"{dataset_name}存在非法标签")      # 存在非法标签时抛出异常

    max_word_length = data["words"].apply(                  # 统计每条数据的词语数量
        lambda words: len(words.split())                    # 按照空格切分并计算长度
    ).max()                                                 # 获取最大词语数量

    if max_word_length > MAX_WORDS:                         # 判断是否正确完成长度截断
        raise ValueError(                                   # 截断失败时抛出异常
            f"{dataset_name}存在超过{MAX_WORDS}词的文本"
        )

    print(f"{dataset_name}字段验证通过")                    # 输出字段验证结果
    print(f"{dataset_name}字段：{actual_columns}")          # 输出字段列表
    print(f"{dataset_name}最大词语数量：{max_word_length}")  # 输出最大词语数量
    print(f"{dataset_name}类别分布：")                      # 输出类别分布提示
    print(data["label"].value_counts().sort_index())        # 输出各类别样本数量


def save_csv_dataset(data, output_path):
    """
    保存处理后的CSV文件。

    参数：
        data：处理后的标准数据集。
        output_path：CSV文件输出路径。
    """

    output_dir = os.path.dirname(output_path)               # 获取输出目录

    if output_dir:                                          # 判断输出目录是否有效
        os.makedirs(                                        # 创建输出目录
            output_dir,                                     # 指定需要创建的目录
            exist_ok=True                                   # 目录存在时不报错
        )

    data = data[                                            # 再次固定最终字段顺序
        ["text", "label", "words"]                          # 指定三个输出字段
    ]

    data.to_csv(                                            # 将数据保存为CSV文件
        output_path,                                        # 指定CSV文件路径
        index=False,                                        # 不保存DataFrame行索引
        encoding="utf-8"                                    # 使用UTF-8编码
    )

    print(f"数据已保存至：{output_path}")                   # 输出文件保存位置


def main():
    """
    组织完整的数据处理流程。
    """

    path_mapping = {                                       # 定义输入文件和输出文件映射
        "训练集": {                                         # 配置训练集路径
            "input": TRAIN_INPUT_PATH,                      # 原始训练集TXT路径
            "output": TRAIN_OUTPUT_PATH                     # 处理后训练集CSV路径
        },
        "验证集": {                                         # 配置验证集路径
            "input": DEV_INPUT_PATH,                        # 原始验证集TXT路径
            "output": DEV_OUTPUT_PATH                       # 处理后验证集CSV路径
        },
        "测试集": {                                         # 配置测试集路径
            "input": TEST_INPUT_PATH,                       # 原始测试集TXT路径
            "output": TEST_OUTPUT_PATH                      # 处理后测试集CSV路径
        }
    }

    for dataset_name, paths in path_mapping.items():       # 依次处理三个数据集
        print("\n" + "=" * 60)                              # 输出分隔线
        print(f"开始处理{dataset_name}")                    # 输出当前数据集名称

        raw_data = read_txt_dataset(                        # 读取原始TXT文件
            paths["input"]                                  # 传入当前输入文件路径
        )

        show_basic_statistics(                             # 输出处理前的基本统计信息
            raw_data,                                      # 传入原始数据
            dataset_name                                   # 传入数据集名称
        )

        processed_data = process_dataset(                  # 执行完整数据处理
            raw_data,                                      # 传入原始数据
            dataset_name                                   # 传入数据集名称
        )

        validate_dataset(                                  # 验证处理后的数据
            processed_data,                                # 传入处理后的数据
            dataset_name                                   # 传入数据集名称
        )

        save_csv_dataset(                                  # 保存标准CSV文件
            processed_data,                                # 传入处理后的数据
            paths["output"]                                # 传入CSV输出路径
        )

    print("\n全部数据处理完成")                             # 输出流程结束提示
    print(f"已生成：{TRAIN_OUTPUT_PATH}")                   # 输出训练集保存路径
    print(f"已生成：{DEV_OUTPUT_PATH}")                     # 输出验证集保存路径
    print(f"已生成：{TEST_OUTPUT_PATH}")                    # 输出测试集保存路径


if __name__ == "__main__":                                 # 判断脚本是否被直接运行
    main()                                                  # 调用主函数执行完整流程
