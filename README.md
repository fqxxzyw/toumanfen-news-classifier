# 投满分项目：中文新闻标题分类

按学习阶段实现新闻十分类。当前版本：**v0.2.0 · 2026-10-18**。

| 阶段 | 教程 | 已实现内容 |
| --- | --- | --- |
| 01 | [数据处理](01_data/README.md) | 分析、清洗、分词、TXT 转 CSV |
| 02 | [随机森林](02_rf/README.md) | TF-IDF、训练、验证、API 和页面，保留 v0.1.0 教程 |
| 03 | [FastText](03_fasttext/README.md) | 字符/词语预处理、训练、调参、四模型评估、预测、API 和页面 |
| 04 | [BERT](04_bert/README.md) | 数据加载、微调训练、dev 指标、早停和最佳模型保存；独立测试与部署待实现 |

查看 [版本记录](CHANGELOG.md)、[本次发布说明](docs/releases/v0.2.0.md) 和
[实际验证记录](docs/VALIDATION.md)。昨日记录保留在 [随机森林 CHANGELOG](02_rf/CHANGELOG.md)。

## 快速运行 FastText

Windows Anaconda Prompt / CMD，从仓库根目录执行；新环境可使用 Python 3.10：

```bat
conda activate toumanfen
cd 03_fasttext
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

本版提供源码、数据和小型配置，**不提供训练模型或大型预训练权重附件**。
已有 `data/processed/` 时可跳过预处理；已有模型时直接放入 `03_fasttext/models/`，跳过训练。
预处理覆盖六个文件；训练会覆盖两个基础模型和两个调参模型，各调参搜索预算为 300 秒。

两个终端均进入 `03_fasttext`，保持运行：

```bat
python src/api.py
```

```bat
python -m streamlit run src/app.py
```

接口 `http://127.0.0.1:8000/predict`；页面 `http://localhost:8501`。
API 默认加载字符自动调参模型，网页调用 API。普通 `python src/app.py` 不会启动网页服务。
当前 Windows fastText 使用相对路径绕过中文绝对路径问题，请从仓库根目录、03_fasttext 或其 src 运行。

## FastText 开发验证结果

| 模型 | dev 准确率 | 样本数 |
| --- | ---: | ---: |
| fasttext_char | 87.17% | 10,000 |
| fasttext_word | 89.90% | 10,000 |
| fasttext_char_autotune | 91.43% | 10,000 |
| fasttext_word_autotune | 91.55% | 10,000 |

dev 同时用于自动调参，表格不能当作最终独立 test 成绩。评估脚本同时输出每类别 P/R/F1/support。

## BERT 当前进度

从仓库根目录执行：

```bat
cd 04_bert
python -m pip install -r requirements.txt
python src/data.py
python src/train.py
```

data.py 只创建并打印 DataLoader；新增的 train.py 执行微调与 dev 验证。
仓库提供分词器配置和词表，训练需要本地另备 PyTorch 预训练权重。
数据加载已验证 180,000 条；训练已完成临时小样本检查，没有自动启动全量训练。
小样本命令、产物与函数说明见 [BERT 教程](04_bert/README.md)。

## 数据与发布约定

- 原始 UTF-8 TXT 每行是 `新闻标题<Tab>整数标签`，train/dev/test 为 180,000/10,000/10,000 条。
- 类别顺序为 finance、realty、stocks、education、science、society、politics、sports、game、entertainment。
- FastText 输入为 `__label__类别 分词文本`，预测时应使用与训练一致的 char/word 模式。
- 模型、BERT 大型权重、IDE、缓存和本地 `预习/` 不上传，文件保留在本地。
- 已有历史中的 CSV 和 FastText 格式数据继续保留，不改写历史。
- 固定版本可下载 GitHub Release 的 Source code (zip)，或使用 `git clone --branch v0.2.0 --depth 1 https://github.com/fqxxzyw/toumanfen-news-classifier.git`。
