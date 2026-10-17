# 投满分项目 · 中文新闻标题十分类

**当前版本：v0.1.0，2026-10-17。** 使用 THUCNews 新闻标题子集，通过 jieba 分词、TF-IDF 特征和随机森林实现十分类，提供 Flask API 与 Streamlit 页面。版本进度见 [CHANGELOG](CHANGELOG.md) 和 [发布说明](docs/releases/v0.1.0.md)。

这是一份从零开始的学习与运行指南。以下命令默认在**项目根目录**执行，每条命令单独输入。Python 环境名称为 `toumanfen`，建议使用 Python 3.10。

## 1. 先了解项目结构

```text
toumanfen-news-classifier/
├── src/
│   ├── 01_eda.py            # 第一步：原始数据探索与绘图
│   ├── 02_data_process.py   # 第二步：清洗、分词、生成 CSV
│   ├── 03_train.py          # 第三步：特征提取、分批训练、保存模型
│   ├── 04_evaluate.py       # 第四步：独立验证集预测与指标导出
│   ├── 05_predict_api.py    # 第五步：启动 Flask 分类接口
│   ├── 06_app.py            # 第六步：启动 Streamlit 交互页面
│   ├── config.py            # 统一的绝对路径配置
│   ├── prediction.py        # 单条新闻预测，供接口导入
│   └── training_utils.py    # 分批增加树、分批预测及进度条
├── data/
│   ├── train.txt            # 原始训练集（180000 条）
│   ├── dev.txt              # 原始验证集（10000 条）
│   ├── test.txt             # 原始测试集（10000 条）
│   ├── class.txt            # 标签编号对应的类别
│   ├── stopwords.txt        # 停用词表
│   └── README.md            # 原始数据来源
├── docs/
│   ├── FILE_MAP.md          # 原文件名与新文件名对照
│   └── releases/v0.1.0.md   # 当前版本的完成情况与限制
├── requirements.txt
├── VERSION
├── CHANGELOG.md
└── README.md
```

运行后会生成 `data/*.csv`、`artifacts/` 和 `model/`，它们不提交到 Git。个人本地的 `预习/` 同样被忽略，不属于远程仓库或 Release 的代码包。

## 2. 安装环境和依赖

其他电脑先获取代码（私有仓库需要先登录有权限的 GitHub 账号）：

```bat
git clone https://github.com/fqxxzyw/toumanfen-news-classifier.git
cd toumanfen-news-classifier
```

第一次使用时，在 Anaconda Prompt 中执行：

```bat
conda create -n toumanfen python=3.10 -y
conda activate toumanfen
```

进入项目目录。下面是本机已有项目的示例；如果上一步已进入克隆目录，不需要再切换：

```bat
cd /d D:\soft\python\NLP\投满分项目
python -m pip install -r requirements.txt
```

如果环境已经存在，只需激活，不要重复创建。PyCharm 解释器选择该环境的 Python，例如 `D:\Anaconda3\envs\toumanfen\python.exe`。

依赖用途：

| 依赖 | 用途 |
| --- | --- |
| pandas、numpy | 表格处理、数值计算与稀疏特征数值类型 |
| matplotlib、seaborn | 类别分布、标题长度分布绘图 |
| jieba | 中文分词 |
| scikit-learn | TF-IDF、随机森林、数据划分与分类指标 |
| tqdm | 训练与预测进度条 |
| Flask | HTTP 分类接口 |
| streamlit | 浏览器交互页面 |
| requests | 页面向接口发送 POST 请求 |

当前 `requirements.txt` 未固定版本；不同 scikit-learn 版本保存的 pickle 模型不保证兼容。更换环境后优先重新训练，并配套使用同一次生成的两个模型文件。

## 3. 理解和准备数据

本项目使用 THUCNews 20 万条中文新闻标题子集，来源及标签表见 [data/README.md](data/README.md)。原始数据保留已有训练、验证、测试划分。

TXT 使用 UTF-8 编码、无表头，每行两列，中间为实际 Tab 制表符：

```text
新闻标题内容<Tab>类别编号
```

`<Tab>` 是格式说明，不是写入数据的字面内容。标签从 `0` 开始，对应 `class.txt` 的行顺序：

| 标签 | 英文类别 | 中文 |
| --- | --- | --- |
| 0 | finance | 财经 |
| 1 | realty | 房产 |
| 2 | stocks | 股票 |
| 3 | education | 教育 |
| 4 | science | 科技 |
| 5 | society | 社会 |
| 6 | politics | 时政 |
| 7 | sports | 体育 |
| 8 | game | 游戏 |
| 9 | entertainment | 娱乐 |

`stopwords.txt` 是分词后需要过滤的词表，按空白字符读取，支持每行一个词。它不会由配置文件自动创建；调整词表后需要重新训练 TF-IDF 和分类器。

## 4. 第一步：探索原始数据

```bat
python src/01_eda.py
```

开发时先读取样本、检查字段与标签，再统计各类样本数、标题长度和重复文本。脚本同时检查不同集合之间的标题交集，便于发现潜在数据泄漏。

生成文件：

- `artifacts/eda/label_distribution.png`：训练集类别分布。
- `artifacts/eda/text_length_distribution.png`：标题长度分布。
- `artifacts/eda/eda_summary.json`：三个数据集的统计及交集检查。

这一步用于了解数据，不训练模型。中文图表显示为方框时，为 Matplotlib 指定本机已安装的中文字体后重新运行。

## 5. 第二步：清洗并分词

```bat
python src/02_data_process.py
```

开发顺序：读取 TXT → 统一字段 → 处理缺失值和非法标签 → 清理 HTML、不可见字符及多余空白 → jieba 分词 → 截断词数 → 校验 → 保存 CSV。

输出 `data/train.csv`、`data/dev.csv`、`data/test.csv`，列顺序固定为：

| 字段 | 含义 |
| --- | --- |
| text | 清洗后的完整标题 |
| label | 整数标签，范围 0～9 |
| words | 空格连接的分词结果，默认最多 30 个词 |

脚本顶部的 `MAX_WORDS=30` 控制最大词数，`NUM_CLASSES=10` 控制标签范围，`DROP_DUPLICATES=False` 表示默认保留集合内部重复样本。修改预处理规则时也应同步检查 `prediction.py` 中在线分词逻辑。

重新执行此步骤会覆盖同名 CSV。

## 6. 第三步：TF-IDF 与随机森林训练

初次运行建议先用小数据确认流程，并将结果放入单独目录：

```bat
python src/03_train.py --limit 2000 --trees 10 --trees-per-batch 2 --output-dir model/quick_check
```

小样本试跑模型不会被正式接口自动使用。正式训练执行：

```bat
python src/03_train.py
```

训练脚本会：

1. 读取 `train.csv`，将其按类别比例划分为 80% 训练部分和 20% 内部评估部分，随机种子为 42。
2. 仅用训练部分拟合 TF-IDF，内部评估部分只执行 `transform`，避免评估文本参与词表和 IDF 学习。
3. 按已有空白分词结果构建特征，保留单字词，使用 `float32` 稀疏矩阵。
4. 通过 `warm_start=True`，每批增加 5 棵树，直到完成 100 棵。
5. 分批预测内部评估部分，输出准确率、宏平均精确率、召回率和 F1。
6. 将分类器和向量化器分别保存到 `model/rf_model.pkl`、`model/tfidf_model.pkl`。

### 训练参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--limit` | 全部训练数据 | 限制读取的样本量，用于快速检查 |
| `--trees` | 100 | 决策树总数 |
| `--trees-per-batch` | 5 | 每批新增的树数；设为 1 可更频繁看到进度更新 |
| `--jobs` | 2 | 并行工作数；内存紧张时设为 1 |
| `--predict-batch-size` | 2000 | 每批预测样本数 |
| `--output-dir` | `model/` | 指定其他模型输出目录 |

随机森林在本项目中按“树”分批，每批仍使用同一完整训练部分；不是按数据块调用 `partial_fit`。训练不会因此变成低内存的流式学习，也不提供中断后的磁盘断点续训。进度条在每批训练完成后更新，首批完成前可能停在 0%。

TF-IDF 提取、模型保存阶段会显示阶段提示，这些阶段没有百分比进度。特征矩阵始终保持稀疏，避免调用整个矩阵的 `toarray()`。

## 7. 第四步：独立验证集评估

```bat
python src/04_evaluate.py
```

加载已保存的两个模型，对原始划分中的 `dev.csv` 进行分批预测，输出宏平均指标和以下文件：

- `artifacts/evaluation/dev_predictions.csv`：文本、真实标签及预测标签。
- `artifacts/evaluation/dev_metrics.json`：样本数量、准确率、精确率、召回率及 F1。

训练脚本中的 20% 内部评估部分与这里的独立 `dev` 集不是同一集合。`test.csv` 留作最终测试，本版本的第四步默认只评估 `dev`，不将未经验证的测试集成绩写成项目结果。

宏平均表示各类别分别计算指标再取平均，不是取最大值。`zero_division=0` 用于处理某类没有预测结果等分母为零的情况。

## 8. 第五步：启动预测接口

确认 `model/` 下已有两个配套模型，然后在第一个终端执行：

```bat
conda activate toumanfen
python src/05_predict_api.py
```

保持终端运行，接口地址为 `http://127.0.0.1:8001/predict`。通过 POST 提交 JSON：

```json
{"text": "中国队在世界锦标赛中夺得冠军"}
```

成功响应结构如下。下面的 `sports` 仅表示响应格式示例，不代表任何输入都保证预测为该类别：

```json
{"result": {"text": "中国队在世界锦标赛中夺得冠军", "label": "sports"}}
```

JSON 不是对象、缺少 `text`、文本不是字符串或只有空白时，返回 HTTP 400。`prediction.py` 负责模型加载、分词、特征转换及标签映射，作为模块供接口使用。Flask 当前使用本地开发服务，不是生产部署方案。

## 9. 第六步：启动网页

第一个终端中的 API 保持运行，在第二个终端执行：

```bat
conda activate toumanfen
python -m streamlit run src/06_app.py
```

打开 `http://localhost:8501`，输入新闻标题，点击“开始分类”。页面通过 `requests` 调用接口，显示中文类别，并处理空输入、连接失败、超时和响应格式异常。

如果终端已经位于 `src/`，请使用：

```bat
python 05_predict_api.py
```

另一个同样位于 `src/` 的终端使用：

```bat
python -m streamlit run 06_app.py
```

不要在 `src/` 内再次写 `src/06_app.py`，也不要将同一命令重复粘贴到一行。

## 10. 如何继续开发

建议按下面的顺序扩展：

1. 记录每次训练参数、依赖版本和验证集指标，建立可复现的实验基线。
2. 根据重复文本检查结果决定是否重新划分数据，先固定评估方案再比较模型。
3. 对比朴素贝叶斯、逻辑回归或线性 SVM 等文本分类基线。
4. 根据独立验证集调参，模型和预处理规则同步更新。
5. 调参完成后单独运行最终测试集评估，保留报告。
6. 补充自动化回归测试、模型版本管理和生产部署。

当前尚未完成上述全部工作，也没有声称提供最新训练实现的全量性能基准。v0.1.0 保留的旧模型与本次整理后的训练代码可能来自不同训练轮次，详见 Release 中的模型说明。

## 11. 文件迁移与常见问题

原文件名、新文件名和职责见 [迁移对照表](docs/FILE_MAP.md)。编号文件是直接运行的入口，共用模块保持普通名称，便于 Python 导入。

- **找不到文件**：先确认数据在项目根目录 `data/` 下。配置根据脚本位置生成绝对路径，通常无需修改 IDE 工作目录。
- **找不到模型**：在新电脑上先运行步骤 2、3；如果已有模型，直接复制配套的两个 `.pkl` 到 `model/`。本机已有模型未重新训练，预习区也已复制好。
- **内存不足或训练慢**：保持稀疏矩阵，先用小样本试跑；将 `--jobs` 设为 1，必要时减少树数。按树分批仍需要保存整片森林。
- **API 提示停用词不一致**：旧模型可能保存了早期向量化配置，需要用当前训练脚本重新生成两个配套模型。
- **网页连接失败**：确认分类接口运行在 8001；Streamlit 页面运行在 8501，二者需要同时启动。
- **改代码后旧进程还在运行**：停止旧服务并重新启动。改文件不会自动改变已经开始的训练进程。
- **本地练习**：`预习/README.md` 给出独立练习步骤，该目录由 `.gitignore` 排除，不会上传。

## 12. 发布与数据说明

代码仓库包含项目代码、原始 TXT 数据、类别表及停用词表，不包含生成的 CSV、模型、IDE 设置或本地预习文件。GitHub Release 提供版本源码包。本轮大型模型附件上传超时，没有发布不完整的附件；现有模型仍保留在本地，且已复制到本地预习区，没有重新训练。

本机现有模型另有归档 `dist/demo-models-v0.1.0.zip` 和 `dist/SHA256SUMS.txt`，均不提交到 Git。模型文件信息见 [MODEL_ARTIFACTS.json](docs/MODEL_ARTIFACTS.json)。

数据及停用词为项目中已有的外部资源。保留其说明，使用时遵循原始来源适用的条款；本项目未额外声明这些资源的授权许可。
