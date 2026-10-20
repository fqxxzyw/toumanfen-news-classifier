# BERT 新闻十分类

完成数据加载、BERT 微调、验证集评估、早停、最优模型保存、独立测试与单条预测。Flask API 与 Streamlit 页面启动方式见 [WEB_README.md](WEB_README.md)。代码中包含详细中文注释。

## 安装与训练

在本目录运行：

```powershell
conda activate toumanfen
python -m pip install -r requirements.txt
python src/data.py
python src/train.py
```

`data.py` 只读取数据并创建 DataLoader；`train.py` 执行训练。PyCharm 可直接运行 `src/train.py`，数据与模型路径按文件位置定位。

训练需要在 `pretrained/bert-base-chinese/` 准备匹配的本地 PyTorch 权重。仓库保留分词器和小型配置文件，不上传大型预训练权重或训练后的模型。

当前训练入口没有命令行超参数解析。需要调整训练设置时修改 `src/config.py`，不要使用旧文档中的 `--epochs`、`--max-train-samples` 或 `--output-dir` 参数。

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `batch_size` | 16 | 每批样本数 |
| `max_length` | 64 | 分词后的最大输入长度 |
| `num_epochs` | 5 | 最多训练轮数 |
| `learning_rate` | 2e-5 | AdamW 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `warmup_ratio` | 0.1 | 学习率预热比例 |
| `patience` | 2 | 连续未提升多少轮后早停 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `seed` | 42 | 随机种子 |
| `num_workers` | 0 | DataLoader 子进程数 |
| `device` | 自动选择 | CUDA 可用时使用 GPU，否则使用 CPU |

## 训练与评估流程

1. 固定随机种子，读取类别名称及其 ID 映射。
2. 加载带分类头的 BERT 和配套 tokenizer，移到运行设备。
3. 训练集打乱，验证集保持顺序；按批次分词、动态补齐、截断并添加 labels。
4. 每批清空梯度，前向计算交叉熵，反向传播，裁剪梯度，更新参数与学习率。
5. 每轮在 dev 集执行 `model.eval()` 与 `torch.no_grad()`，汇总真实标签和预测标签。
6. 验证集宏平均 F1 严格提升时保存模型；否则连续未提升次数加一，达到 patience 后停止。

`metrics.py` 的 `cal_metrics` 返回 `(accuracy, precision, recall, f1, report)` 元组，训练代码使用下标 3 读取 F1。P/R/F1 按实际出现的真实或预测类别做宏平均；分类报告显式列出全部类别。仅用于展示的训练损失是各批平均损失的均值。

当前入口会更新相同的 best_model 目录，不含非空目录保护、历史日志自动生成或断点续训。重新实验前需要保留旧模型时，请先自行备份目录或修改配置中的保存路径。

## 独立测试与单条预测

先确保存在完整的 `models/best_model`：

```powershell
python src/test_model.py
python src/predictor.py
```

`test_model.py` 自动加载正确的 BERT 模型类型，并读取 `config.test_path` 评估测试集；它与训练共用评估函数。`predictor.py` 运行文件末尾的示例文本，可修改后执行，也可在其他脚本导入：

```python
from predictor import BertNewsClassifierPredictor

predictor = BertNewsClassifierPredictor()
result = predictor.predict_one("球队在总决赛中获得冠军")
print(result)
```

返回 `pred`（类别 ID）、`class`（英文标签）和 `score`（该类别的 softmax 概率）。分数不等于测试集准确率；单条预测器加载一次后可重复使用。

## API 与页面

两个终端均位于 `04_bert`，分别执行：

```powershell
python src/api.py
```

```powershell
python -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8501
```

页面为 <http://127.0.0.1:8501>，接口为 `POST http://127.0.0.1:8000/predict`，请求体 `{"text": "新闻标题"}`。页面通过 HTTP 调用 API，模型只在 API 启动时加载一次。与 FastText 的默认端口重复，切换时请先停止旧服务。更多调用示例、错误码和分享说明见 [WEB_README.md](WEB_README.md)。

## 保存内容与分享

训练时保存至 `models/best_model/`：

- `model.safetensors` 等权重与 `config.json`：完整微调模型及类别映射。
- tokenizer 文件：配套分词器。
- `metadata.json`：类别顺序、最大输入长度、最优验证集 F1 和训练来源路径。

当前脚本不生成 `history.json`、`best_metrics.json` 或 `training_args.json`；如果本地目录有这些文件，可能是先前实现留下的记录。

给同学做预测时，提供整个 best_model 文件夹与预测代码、运行依赖即可，不需要原始预训练模型。约 390 MB 的权重不在 GitHub 源码中，需要另行提供；metadata 中记录的原始训练路径不要求在对方电脑上存在。

## 数据与函数

原始数据每行 UTF-8 `新闻标题<Tab>整数标签`，train/dev/test 分别为 180,000/10,000/10,000 条。类别 ID 对应 `class.txt` 的顺序。

| 函数或类 | 用途 |
| --- | --- |
| `load_class_list()` | 按 ID 顺序读取类别名称 |
| `load_news_data(file_path, max_samples=None)` | 读取文本与整数标签，支持可选样本数量限制 |
| `NewDataSet` | 保存样本，提供长度与下标取样 |
| `build_context_fn(tokenizer, max_length=None)` | 返回分词、动态补齐、添加 labels 的批处理函数 |
| `build_dataloader(file_path, tokenizer, shuffle, max_samples=None, config=None)` | 构建 DataLoader，首个参数是文件路径，配置通过 config= 单独传入 |
| `cal_metrics(y_true, y_pred, class_name)` | 计算指标和逐类别报告 |
| `evaluate(model, data_loader, device, class_name)` | 无梯度评估并返回指标元组 |
| `train()` | 组织训练、验证、保存与早停 |
| `BertNewsClassifierPredictor.predict_one(text)` | 单条文本预测 |

## 验证

```powershell
python -m unittest discover -s tests -p test_web.py -v
```

六项检查验证 API 输入处理及页面交互，不加载真实 BERT。开发时另用本地真实模型验证了单条预测、前 16 条测试样本以及页面到 Flask 的完整调用，详见 [验证记录](../docs/VALIDATION.md)。这些检查不构成全量 test 基准；没有因整理源码而重新执行完整训练。
