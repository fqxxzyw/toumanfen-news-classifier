"""步骤 01：分析原始数据，生成 EDA 报告。"""
from pathlib import Path  # 导入Path，用于统一管理项目路径
import json               # 导入JSON模块，用于保存EDA报告

import matplotlib.pyplot as plt  # 导入Matplotlib绘图工具
import numpy as np                # 导入NumPy统计工具
import pandas as pd               # 导入Pandas数据分析工具
import seaborn as sns             # 导入Seaborn统计绘图库


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录
DATA_DIR = PROJECT_ROOT / "data"                    # 定义原始数据目录
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"     # 定义EDA结果目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)       # 创建输出目录

CLASS_PATH = DATA_DIR / "class.txt"                 # 定义类别文件路径

SPLIT_PATHS = {  # 创建字典，用来保存训练集、验证集和测试集的文件路径
    "train": DATA_DIR / "train.txt",                # 定义训练集路径
    "dev": DATA_DIR / "dev.txt",                    # 定义验证集路径
    "test": DATA_DIR / "test.txt",                  # 定义测试集路径
}  # 定义三个数据集的名称和路径

plt.rcParams['axes.unicode_minus'] = False

def load_classes(path: Path) -> list[str]:  # 定义 load_classes 函数：输入类别文件路径，返回类别名称列表
    """读取class.txt中的类别名称。"""  # 函数说明：读取 class.txt 中的所有类别名称

    if not path.exists():                            # 检查类别文件是否存在
        raise FileNotFoundError(f"类别文件不存在：{path}")  # 如果文件不存在，主动抛出异常并提示具体路径

    classes = [  # 开始使用列表推导式构造类别名称列表
        line.strip()  # 去掉当前类别名称前后的空格和换行符
        for line in path.read_text(encoding="utf-8").splitlines()  # 读取整个类别文件，并按行拆分后逐行遍历
        if line.strip()  # 只保留去除空白后仍然有内容的行
    ]  # 读取非空类别名称

    if not classes:                                  # 检查类别列表是否为空
        raise ValueError("class.txt中没有有效类别")  # 如果没有读取到任何有效类别，则抛出异常

    if len(classes) != len(set(classes)):            # 检查类别名称是否重复
        raise ValueError("class.txt中存在重复类别")  # 如果类别名称出现重复，则抛出异常，避免标签映射混乱

    return classes                                   # 返回类别名称列表

"""
实现加载数据集的功能
具体的思路
1. 判断数据是否存在
2. 加载数据
3. 数据清洗(去除左右空白）并将标签转换成数字
4. 对没法转换成数字的标签进行异常抛出
5. 判断标签是否有越界行为，如果有越界行为就抛出异常

半古法，半AI
"""

def load_dataset(path: Path, num_classes: int) -> pd.DataFrame:  # 定义 load_dataset 函数：读取一个数据集文件并做基础合法性检查
    """读取文本分类数据并完成基础格式检查。"""  # 函数说明：读取文本分类数据，并检查标签格式和范围

    if not path.exists():                            # 检查数据文件是否存在
        raise FileNotFoundError(f"数据文件不存在：{path}")  # 如果数据文件不存在，则抛出异常并提示具体文件路径

    df = pd.read_csv(  # 调用 Pandas 的 read_csv 读取文本分类数据
        path,                                       # 指定数据文件路径
        sep="\t",                                   # 指定Tab分隔符
        names=["sentence", "label"],                 # 指定字段名称
        header=None,                                # 原始文件没有表头
        encoding="utf-8",                           # 使用UTF-8编码
        keep_default_na=False,                      # 保留原始空字符串
    )  # 读取数据文件

    df["sentence"] = df["sentence"].astype(str).str.strip()  # 去除文本首尾空白

    numeric_labels = pd.to_numeric(  # 调用 pd.to_numeric，尝试把原始标签转换为数值类型
        df["label"],  # 指定需要进行数值转换的 label 列
        errors="coerce",  # 无法转换的值不直接报错，而是先转成 NaN 便于后续检查
    )  # 尝试把标签转换为数字

    invalid_number_mask = numeric_labels.isna()      # 找出无法转换为数字的标签

    if invalid_number_mask.any():                   # 如果存在非数字标签
        bad_rows = df.loc[  # 从原始数据中筛选出标签无法转换成数字的异常样本
            invalid_number_mask,  # 使用布尔掩码定位所有标签格式错误的行
            ["sentence", "label"],  # 只保留 sentence 和 label 两列，方便查看问题数据
        ].head(5)  # 最多取前 5 条异常样本，避免报错信息过长

        raise ValueError(  # 如果发现非数字标签，则准备抛出 ValueError
            f"{path.name}存在非数字标签：\n{bad_rows}"  # 在异常信息中显示文件名以及部分错误样本
        )  # 结束 ValueError 的参数列表

    df["label"] = numeric_labels.astype(int)         # 将合法标签转换为整数

    invalid_range_mask = ~df["label"].between(  # 检查标签编号是否落在合法类别编号范围内
        0,  # 合法标签的最小值固定为 0
        num_classes - 1,  # 合法标签的最大值为 类别数量减 1
    )  # 找出超出类别范围的标签

    if invalid_range_mask.any():                    # 如果存在越界标签
        bad_labels = sorted(  # 收集所有越界的标签编号，便于定位数据问题
            df.loc[invalid_range_mask, "label"].unique()  # 筛选越界标签、去重后得到实际出现的非法编号
        )  # 结束 sorted 函数调用

        raise ValueError(  # 如果存在越界标签，则准备抛出 ValueError
            f"{path.name}存在越界标签：{bad_labels}"  # 在异常信息中显示数据文件名和所有非法标签编号
        )  # 结束 ValueError 的参数列表

    return df  # 检查通过后，返回读取完成的数据

def analyze_split(  # 定义 analyze_split 函数：分析 train、dev 或 test 中的一个数据集
    split_name: str,  # split_name 表示当前数据集名称，例如 train、dev 或 test
    df: pd.DataFrame,  # df 表示当前数据集对应的 Pandas DataFrame
    classes: list[str],  # classes 表示完整类别名称列表
) -> dict:  # 声明函数返回一个 dict 类型的统计结果
    """分析单个数据集，并返回可保存的统计结果。"""  # 函数说明：统计单个数据集的样本、类别、长度和异常情况

    df = df.copy()                                  # 复制数据，避免修改原始DataFrame
    df["sentence_len"] = df["sentence"].str.len()   # 计算每条文本的字符长度

    label_counts = (  # 开始统计每一个类别分别有多少条样本
        df["label"]  # 先取出数据集中的 label 标签列
        .value_counts()  # 统计每个不同标签出现的次数
        .sort_index()  # 按照标签编号从小到大排序
        .reindex(range(len(classes)), fill_value=0)  # 补齐没有出现的类别，并把这些类别的样本数填为 0
    )  # 按照固定标签顺序统计类别数量

    label_distribution = []                        # 创建类别分布结果列表

    for label_id, class_name in enumerate(classes):  # 依次遍历类别编号和类别名称
        count = int(label_counts.loc[label_id])     # 获取当前类别样本数量
        ratio = count / len(df) if len(df) else 0   # 计算当前类别占比

        label_distribution.append(  # 把当前类别的统计结果追加到 label_distribution 列表中
            {  # 开始构造当前类别的统计字典
                "label_id": label_id,  # 记录当前类别的数字编号
                "class_name": class_name,  # 记录当前类别对应的文本名称
                "count": count,  # 记录当前类别的样本数量
                "ratio": ratio,  # 记录当前类别在整个数据集中的占比
            }  # 结束当前类别统计字典
        )  # 保存当前类别统计结果

    label_count_per_text = df.groupby(  # 按照 sentence 文本内容进行分组
        "sentence"  # 指定 sentence 作为分组键，也就是把相同文本放到一组
    )["label"].nunique()                            # 统计相同文本对应的标签数量

    conflict_count = int(  # 开始计算“同一文本对应多个标签”的文本数量
        (label_count_per_text > 1).sum()  # 统计标签种类数大于 1 的文本有多少条
    )  # 统计同文异标数量

    length_values = df["sentence_len"]              # 获取文本长度序列

    summary = {  # 开始构造当前数据集的汇总统计字典
        "split": split_name,  # 记录当前数据集名称
        "sample_count": int(len(df)),  # 记录当前数据集的总样本数量
        "class_count": int(df["label"].nunique()),  # 记录当前数据集中实际出现了多少个类别
        "empty_sentence_count": int(  # 开始统计空文本数量
            (df["sentence"] == "").sum()  # 判断 sentence 是否为空字符串，并统计满足条件的数量
        ),  # 结束空文本数量的 int 转换
        "duplicate_row_count": int(  # 开始统计“文本和标签都完全相同”的重复行数量
            df.duplicated(["sentence", "label"]).sum()  # 根据 sentence 和 label 两列判断完全重复记录
        ),  # 结束完全重复数量的 int 转换
        "duplicate_text_count": int(  # 开始统计只看文本内容时的重复文本数量
            df.duplicated(["sentence"]).sum()  # 只根据 sentence 列判断重复文本
        ),  # 结束重复文本数量的 int 转换
        "conflict_text_count": conflict_count,  # 记录前面已经算出的同文异标数量
        "garbled_text_count": int(  # 开始统计疑似乱码文本的数量
            df["sentence"].str.contains(  # 检查 sentence 中是否包含 Unicode 替换字符“�”
                "�",  # 指定要查找的乱码替换字符
                regex=False,  # 关闭正则表达式匹配，把“�”当普通字符处理
            ).sum()  # 对匹配结果求和，得到疑似乱码样本数量
        ),  # 结束乱码数量的 int 转换
        "length": {  # 开始保存文本长度相关统计指标
            "min": int(length_values.min()),  # 记录最短文本的字符数量
            "mean": float(length_values.mean()),  # 记录文本长度的平均值
            "variance": float(length_values.var(ddof=0)),  # 记录文本长度总体方差，ddof=0 表示按总体方差计算
            "std": float(length_values.std(ddof=0)),  # 记录文本长度总体标准差，ddof=0 表示按总体标准差计算
            "p50": float(length_values.quantile(0.50)),  # 记录文本长度的 P50，也就是中位数
            "p90": float(length_values.quantile(0.90)),  # 记录文本长度的 P90，90% 的文本长度不超过该值
            "p95": float(length_values.quantile(0.95)),  # 记录文本长度的 P95，95% 的文本长度不超过该值
            "p99": float(length_values.quantile(0.99)),  # 记录文本长度的 P99，99% 的文本长度不超过该值
            "max": int(length_values.max()),  # 记录最长文本的字符数量
        },  # 结束 length 文本长度统计子字典
        "label_distribution": label_distribution,  # 把前面生成的类别分布明细保存到汇总结果中
    }  # 构造当前数据集的完整统计结果

    print(f"\n===== {split_name} =====")  # 输出数据集名称
    print("样本总量：", summary["sample_count"])  # 输出样本总量
    print("类别数量：", summary["class_count"])  # 输出实际类别数量
    print("空文本数量：", summary["empty_sentence_count"])  # 打印当前数据集中的空文本数量
    print("完全重复数量：", summary["duplicate_row_count"])  # 打印文本和标签完全相同的重复记录数量
    print("重复文本数量：", summary["duplicate_text_count"])  # 打印仅文本内容重复的样本数量
    print("同文异标数量：", summary["conflict_text_count"])  # 打印相同文本对应不同标签的数量
    print("疑似乱码数量：", summary["garbled_text_count"])  # 打印疑似出现乱码的文本数量
    print("文本长度统计：", summary["length"])  # 打印文本长度的全部统计指标

    for item in label_distribution:  # 遍历类别分布结果
        print(  # 开始分多行打印当前类别的编号、名称、数量和占比
            f"{item['label_id']:>2} "  # 输出类别编号，并使用 >2 设置至少 2 个字符宽度右对齐
            f"{item['class_name']:<14} "  # 输出类别名称，并使用 <14 设置至少 14 个字符宽度左对齐
            f"{item['count']:>7} "  # 输出类别样本数量，并使用 >7 设置至少 7 个字符宽度右对齐
            f"({item['ratio']:.2%})"  # 输出类别占比，并使用 .2% 格式化为保留 2 位小数的百分比
        )  # 输出类别ID、类别名称、数量和占比

    return summary  # 返回统计结果



def draw_label_distribution(  # 定义 draw_label_distribution 函数：绘制训练集类别分布图
    train_df: pd.DataFrame,  # train_df 表示训练集 DataFrame
    classes: list[str],  # classes 表示所有类别名称列表
) -> None:  # 该函数只负责绘图，因此返回值为 None
    """绘制训练集类别分布图。"""  # 函数说明：生成训练集不同类别的样本数量柱状图

    label_counts = (  # 开始统计训练集中每个类别的样本数量
        train_df["label"]  # 先取出训练集中的 label 标签列
        .value_counts()  # 统计每个标签编号出现的次数
        .sort_index()  # 按照标签编号从小到大排序
        .reindex(range(len(classes)), fill_value=0)  # 补齐没有样本的类别，并把对应数量填为 0
    )  # 统计每个类别的样本数量

    plot_df = pd.DataFrame(  # 把类别名称和样本数量整理成适合 Seaborn 绘图的 DataFrame
        {  # 开始构造绘图数据字典
            "class_name": classes,  # class_name 列保存每个类别的名称
            "count": label_counts.values,  # count 列保存每个类别对应的样本数量
        }  # 结束绘图数据字典
    )  # 构造绘图数据

    plt.figure(figsize=(12, 6))                    # 创建画布

    sns.barplot(  # 调用 Seaborn 的 barplot 绘制柱状图
        data=plot_df,  # 指定刚才整理好的 plot_df 作为绘图数据源
        x="class_name",  # 横轴使用 class_name，也就是新闻类别名称
        y="count",  # 纵轴使用 count，也就是每个类别的样本数量
        color="#4C78A8",  # 设置柱子的填充颜色为蓝色
    )  # 绘制类别分布柱状图

    plt.title("训练集类别分布")                     # 设置标题
    plt.xlabel("新闻类别")                          # 设置横轴名称
    plt.ylabel("样本数量")                          # 设置纵轴名称
    plt.xticks(rotation=45)                         # 旋转类别名称
    plt.tight_layout()                              # 自动调整边距

    plt.savefig(  # 调用 savefig 把类别分布图保存到文件
        OUTPUT_DIR / "label_distribution.png",  # 指定图片保存路径和文件名 label_distribution.png
        dpi=160,  # 设置导出图片分辨率为 160 DPI
    )  # 保存类别分布图

    plt.close()


def draw_text_length_distribution(  # 定义 draw_text_length_distribution 函数：绘制文本长度分布图
    train_df: pd.DataFrame,  # train_df 表示需要分析的训练集 DataFrame
) -> None:  # 该函数只负责绘图，因此返回值为 None
    """绘制训练集文本长度分布图。"""  # 函数说明：生成训练集文本长度直方图并标记 P95

    lengths = train_df["sentence"].str.len()        # 计算训练集文本长度
    p95 = lengths.quantile(0.95)                    # 计算文本长度P95

    plt.figure(figsize=(12, 6))                    # 创建画布

    sns.histplot(  # 调用 Seaborn 的 histplot 绘制文本长度直方图
        lengths,  # 把每条文本的字符长度作为直方图输入数据
        bins=30,  # 把长度范围划分为 30 个区间进行统计
        color="#59A14F",  # 设置直方图柱子的填充颜色为绿色
    )  # 绘制文本长度直方图

    plt.axvline(  # 调用 axvline 在图上绘制一条竖直参考线
        p95,  # 参考线的位置设置为前面计算出的 P95 文本长度
        color="red",  # 把参考线颜色设置为红色
        linestyle="--",  # 把参考线设置为虚线样式
        label=f"P95={p95:.0f}",  # 设置图例文字，例如 P95=128
    )  # 使用红色虚线标记P95

    plt.title("训练集文本长度分布")                 # 设置标题
    plt.xlabel("字符数量")                          # 设置横轴名称
    plt.ylabel("样本数量")                          # 设置纵轴名称
    plt.legend()                                    # 显示P95图例
    plt.tight_layout()                              # 自动调整边距

    plt.savefig(  # 调用 savefig 把文本长度分布图保存到文件
        OUTPUT_DIR / "text_length_distribution.png",  # 指定图片保存路径和文件名 text_length_distribution.png
        dpi=160,  # 设置导出图片分辨率为 160 DPI
    )  # 保存文本长度分布图

    plt.close()                                     # 关闭画布


def check_split_overlap(  # 定义 check_split_overlap 函数：检查不同数据集之间是否存在重复文本
    datasets: dict[str, pd.DataFrame],  # datasets 是一个字典，键为数据集名称，值为对应 DataFrame
) -> dict:  # 声明函数返回一个 dict 类型的交集检查结果
    """检查训练集、验证集和测试集之间的重复文本。"""  # 函数说明：检查 train、dev、test 之间是否出现相同文本

    text_sets = {  # 开始把每个数据集中的文本转换成集合，方便做集合交集运算
        split_name: set(df["sentence"])  # 把当前数据集的 sentence 列转换为 set，自动去除内部重复文本
        for split_name, df in datasets.items()  # 遍历 datasets 中的每一个数据集名称和 DataFrame
    }  # 将每个数据集的文本转换为集合

    split_pairs = [  # 定义需要两两检查的三个数据集组合
        ("train", "dev"),  # 检查训练集和验证集之间是否有重复文本
        ("train", "test"),  # 检查训练集和测试集之间是否有重复文本
        ("dev", "test"),  # 检查验证集和测试集之间是否有重复文本
    ]  # 定义需要检查的数据集组合

    overlap_summary = {}                            # 创建交集统计结果

    for left_name, right_name in split_pairs:  # 依次遍历每一组需要检查的数据集名称
        overlap = (  # 开始计算当前两个数据集的文本交集
            text_sets[left_name]  # 取左侧数据集的文本集合
            & text_sets[right_name]  # 使用 & 集合运算符与右侧文本集合求交集
        )  # 计算两个数据集的文本交集

        key = f"{left_name}_{right_name}"            # 构造结果字段名称

        overlap_summary[key] = {  # 开始保存当前这一组数据集的交集统计结果
            "count": len(overlap),  # 记录两个数据集之间重复文本的总数量
            "examples": list(overlap)[:5],  # 最多保存前 5 条重复文本作为示例
        }  # 保存交集数量和前5条示例

        print(  # 开始分多行打印当前两个数据集的重复文本数量
            f"{left_name}/{right_name}"  # 输出数据集组合名称，例如 train/dev
            f"重复文本数量：{len(overlap)}"  # 紧接着输出这两个数据集之间的重复文本数量
        )  # 输出重复文本数量

        if overlap:                                 # 如果存在重复文本
            print("示例：", list(overlap)[:5])       # 输出部分重复示例

    return overlap_summary                          # 返回交集统计结果


def main() -> None:  # 定义 main 主函数：按顺序组织整个 EDA 执行流程
    """组织完整EDA流程。"""  # 函数说明：负责调用前面定义的所有读取、分析和绘图函数

    classes = load_classes(CLASS_PATH)              # 读取类别名称
    print("类别列表：", classes)                     # 输出类别列表

    datasets = {  # 开始一次性读取 train、dev、test 三个数据集
        split_name: load_dataset(  # 调用 load_dataset 读取当前数据集文件
            path,  # 把当前循环中的文件路径传给 load_dataset
            num_classes=len(classes),  # 把类别总数传入，用来检查标签编号是否越界
        )  # 结束当前一次 load_dataset 函数调用
        for split_name, path in SPLIT_PATHS.items()  # 遍历 SPLIT_PATHS，把三个数据集分别读入 datasets 字典
    }  # 读取训练集、验证集和测试集

    summary = {  # 开始创建最终要保存的完整 EDA 汇总报告
        "classes": classes,  # 保存完整类别名称列表
        "splits": {},  # 预留 splits 字典，用于保存 train、dev、test 各自的分析结果
        "overlap": {},  # 预留 overlap 字典，用于保存数据集之间的重复文本检查结果
    }  # 创建完整EDA报告

    for split_name, df in datasets.items():  # 依次遍历已经读取好的 train、dev、test 数据集
        summary["splits"][split_name] = analyze_split(  # 调用 analyze_split 分析当前数据集，并把结果保存到 summary
            split_name,  # 传入当前数据集名称，例如 train
            df,  # 传入当前数据集对应的 DataFrame
            classes,  # 传入完整类别列表，用于类别统计和补齐缺失类别
        )  # 分析每个数据集并保存结果

    summary["overlap"] = check_split_overlap(  # 调用 check_split_overlap 检查三个数据集之间是否存在文本泄漏
        datasets  # 把已经读取好的所有数据集传给交集检查函数
    )  # 检查不同数据集之间的重复文本

    draw_label_distribution(  # 调用 draw_label_distribution 绘制训练集类别分布图
        datasets["train"],  # 把训练集 DataFrame 传给绘图函数
        classes,  # 传入完整类别列表，保证横轴类别顺序固定
    )  # 绘制训练集类别分布图

    draw_text_length_distribution(  # 调用 draw_text_length_distribution 绘制训练集文本长度分布图
        datasets["train"]  # 把训练集 DataFrame 传给文本长度绘图函数
    )  # 绘制训练集文本长度分布图

    summary_path = OUTPUT_DIR / "eda_summary.json"  # 定义JSON报告路径

    summary_path.write_text(  # 调用 Path.write_text，把最终 EDA 报告写入 JSON 文件
        json.dumps(  # 先调用 json.dumps，把 Python 字典转换成 JSON 字符串
            summary,  # 把 summary 完整统计结果作为需要序列化的数据
            ensure_ascii=False,  # 设置 ensure_ascii=False，让中文直接保存而不是变成 Unicode 转义
            indent=2,  # 设置缩进为 2 个空格，让生成的 JSON 更容易阅读
        ),  # 结束 json.dumps 函数调用
        encoding="utf-8",  # 指定使用 UTF-8 编码写入 JSON 文件
    )  # 保存完整EDA报告

    print("\nEDA执行完成")  # 打印提示信息，说明整个 EDA 流程已经执行完成
    print("类别分布图：", OUTPUT_DIR / "label_distribution.png")  # 打印类别分布图最终保存的位置
    print("长度分布图：", OUTPUT_DIR / "text_length_distribution.png")  # 打印文本长度分布图最终保存的位置
    print("统计报告：", summary_path)  # 打印 JSON 统计报告最终保存的位置


if __name__ == "__main__":  # 只有直接运行当前 Python 文件时，才执行下面的 main 函数
    main()  # 执行完整EDA流程
