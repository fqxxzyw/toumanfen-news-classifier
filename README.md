# 投满分项目：中文新闻标题分类

按学习阶段实现新闻十分类。当前版本：**v0.3.0 · 2026-10-20**。

| 阶段 | 教程 | 已实现内容 |
| --- | --- | --- |
| 01 | [数据处理](01_data/README.md) | 分析、清洗、分词、TXT 转 CSV |
| 02 | [随机森林](02_rf/README.md) | TF-IDF、训练、验证、API 和页面，保留 v0.1.0 教程 |
| 03 | [FastText](03_fasttext/README.md) | 字符/词语预处理、训练、调参、四模型评估、预测、API 和页面 |
| 04 | [BERT](04_bert/README.md) | 微调、dev 验证、早停、独立测试、单条预测、Flask API 和 Streamlit 页面 |
| 05 | [DeepSeek / 大肥鱼](05_LLM/README.md) | LLM 新闻十分类、命令行交互、JSON 校验和大肥鱼主题 Streamlit 页面 |

查看 [版本记录](CHANGELOG.md)、[本次发布说明](docs/releases/v0.3.0.md)、
[今日进度](docs/progress/2026-10-20.md) 和 [实际验证记录](docs/VALIDATION.md)。早期记录保留在 [随机森林 CHANGELOG](02_rf/CHANGELOG.md)。

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

## 运行 BERT

从仓库根目录执行：

```bat
cd 04_bert
python -m pip install -r requirements.txt
python src/data.py
python src/train.py
```

data.py 创建并打印 DataLoader；train.py 执行微调与 dev 验证。
仓库提供分词器配置和词表，训练需要本地另备 PyTorch 预训练权重。
训练参数在 `04_bert/src/config.py` 中修改；当前训练入口不解析命令行超参数。
训练后运行 `python src/test_model.py` 评估 test 集，或运行 `python src/predictor.py` 进行单条预测。

有完整 `models/best_model` 后，在两个终端分别启动：

```bat
python src/api.py
```

```bat
python -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
```

BERT API 使用 8000 端口，页面使用 8501 端口；与 FastText 的默认端口重复，切换项目时请先停止旧服务。
详见 [BERT 教程](04_bert/README.md) 和 [API / 页面说明](04_bert/WEB_README.md)。

## 运行 DeepSeek / 大肥鱼主题页面

从仓库根目录执行：

```powershell
conda activate toumanfen
cd 05_LLM
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

首次配置时复制样例；若本地已有 `.env`，请直接编辑原文件。仅在本机填写 `DEEPSEEK_API_KEY`，不要把密钥提交到仓库。

```powershell
# 交互分类，输入 q 退出
python src/deepseek_client.py

# 单条分类，可用 --save 追加保存结果
python src/deepseek_client.py --text "球队在总决赛中获得冠军"

# 大肥鱼主题页面
python -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8502
```

页面位于 `http://127.0.0.1:8502`，也可以在侧栏输入仅限本次会话使用的密钥。
输出包含十分类标签与理由；客户端校验 JSON 和类别范围。实现与依赖详见 [DeepSeek 教程](05_LLM/README.md)。

## 今日验证

- BERT：6 项 API / 页面检查通过；已用本地真实模型验证预测、16 条测试样本评估和页面到 API 的完整调用。
- DeepSeek：6 项离线检查通过，覆盖单条/交互命令行、非法返回值、缺少密钥和页面分类。
- DeepSeek 离线检查不调用外部 API；真实分类需要在本地配置有效密钥与账户额度。
- 验证未重新执行完整 BERT 训练或全量 test 基准，不把小样本结果作为完整成绩。

验证详情见 [验证记录](docs/VALIDATION.md)。

## 数据与发布约定

- 原始 UTF-8 TXT 每行是 `新闻标题<Tab>整数标签`，train/dev/test 为 180,000/10,000/10,000 条。
- 类别顺序为 finance、realty、stocks、education、science、society、politics、sports、game、entertainment。
- FastText 输入为 `__label__类别 分词文本`，预测时应使用与训练一致的 char/word 模式。
- 模型、BERT 大型权重、IDE、缓存和本地 `预习/` 不上传，文件保留在本地。
- 密钥配置只提交 `05_LLM/.env.example` 空值样例；`.env`、`.env.local` 与真实密钥不上传。
- 大肥鱼页面使用项目内的主题插画，可离线加载；图片来源与生成提示词见 [资源说明](05_LLM/assets/README.md)。
- 已有历史中的 CSV 和 FastText 格式数据继续保留，不改写历史。
- 固定版本可下载 [v0.3.0 Release](https://github.com/fqxxzyw/toumanfen-news-classifier/releases/tag/v0.3.0) 的 Source code (zip)，或使用 `git clone --branch v0.3.0 --depth 1 https://github.com/fqxxzyw/toumanfen-news-classifier.git`。
