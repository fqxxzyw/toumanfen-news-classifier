# 01：原始数据到 CSV

本目录专门保存数据分析和数据预处理过程。代码从 `../02_rf/src/` 原样复制，
保留原来的实现和注释，不包含模型训练、接口或页面代码。

```text
01_data/
├── data/
│   ├── train.txt
│   ├── dev.txt
│   ├── test.txt
│   ├── class.txt
│   ├── stopwords.txt
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   └── README.md
├── src/
│   ├── 01_eda.py
│   └── 02_data_process.py
└── README.md
```

- `01_eda.py`：读取原始 TXT，检查类别、文本长度、数据质量和数据集交叉重复，结果写入本目录的 `artifacts/eda/`。
- `02_data_process.py`：读取原始 TXT，清洗文本、检查标签、jieba 分词、最多保留 30 个词，导出 CSV。
- CSV 字段依次为 `text`、`label`、`words`。train/dev/test 分别处理，不混合。

TXT、CSV、类别表和停用词均从 `02_rf/data/` 复制，已有 CSV 不是本次重新生成的。
脚本使用自身路径定位本目录的 `data/`；数据处理脚本不依赖 `config.py`，因此没有复制模型配置。

以后需要手动重跑时，在本目录执行以下命令（本次整理没有执行）：

```bat
python src/01_eda.py
python src/02_data_process.py
```

依赖可参考 `../02_rf/requirements.txt`，这里主要使用 pandas、numpy、jieba、matplotlib、seaborn。
重跑会写入本目录的结果，不会自动同步到 `02_rf/data/`。
