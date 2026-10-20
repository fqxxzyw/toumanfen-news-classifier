# BERT 新闻分类训练

当前实现配置、数据加载、十分类微调、dev 验证、宏平均指标、早停和最佳模型保存。
后续独立 test_model.py、predictor.py 和 Flask 服务尚未实现。

## 运行

从本目录执行：

```bat
conda activate toumanfen
python -m pip install -r requirements.txt
python src/data.py
python src/train.py
```

data.py 只打印类别并创建 DataLoader；train.py 才启动训练。
也可以在 PyCharm 中直接运行 src/train.py，路径按脚本位置定位。

## 先用小样本试跑

```bat
python src/train.py --epochs 1 --batch-size 2 --max-train-samples 32 --max-dev-samples 16 --output-dir models/smoke_test
```

样本限制读取文件前若干条，只用于检查流程，不是正式成绩。
正式训练直接运行 `python src/train.py`，默认使用全部 train/dev 数据、batch_size=16、5 轮。
当前 toumanfen 解释器为 CPU 版 PyTorch；脚本自动选择设备，CPU 全量训练耗时较长。
可以使用 --device cpu/cuda，显式指定 cuda 时会检查是否可用。

## 训练流程与配置

1. 固定随机种子，读取十个类别和本地预训练权重，初始化新闻分类头。
2. 训练集打乱，dev 不打乱；训练和验证均复用 data.py 的分词与动态补齐。
3. 每轮训练：前向计算交叉熵、反向传播、梯度裁剪、AdamW 更新、学习率更新。
4. dev 阶段使用 eval 和 inference_mode，计算平均 loss、accuracy、宏平均 P/R/F1。
5. dev 宏平均 F1 严格提升才保存最佳模型；连续 patience 轮未提升则早停。
6. 每轮更新 history.json，显示进度条、loss、accuracy 和学习率；不读取 test 集。

| 参数 | 默认值 | 修改方式 |
| --- | --- | --- |
| epochs | 5 | --epochs |
| batch_size | 16 | --batch-size |
| max_length | 64 | --max-length |
| learning_rate | 2e-5 | --learning-rate |
| patience | 2 | --patience |
| weight_decay | 0.01 | config.py |
| warmup_ratio | 0.1 | config.py；预热后线性衰减 |
| max_grad_norm | 1.0 | config.py |
| seed | 42 | config.py |

--pretrained-dir 指定本地预训练目录，--output-dir 指定训练结果目录。
默认不覆盖非空输出目录；重新实验可指定新目录，确实要覆盖时使用 --overwrite-output。
当前保存最佳推理权重，不保存优化器/调度器状态，尚不支持中断后继续原训练步数。

## 输出文件

默认保存到 `04_bert/models/best_model/`，模型目录已被 Git 忽略：

- model.safetensors 等权重文件及 config.json：最佳十分类模型与类别映射。
- tokenizer 配置和词表：与最佳模型配套的分词器。
- best_metrics.json：最佳轮次、dev loss、accuracy、宏平均 P/R/F1。
- history.json：每轮训练/验证指标、学习率、耗时和是否刷新最佳值。
- training_args.json：本次路径、参数、类别顺序、随机种子与模型选择指标。

后续测试或部署时，模型与分词器都应从这个目录加载。
首次把预训练模型转成分类模型时，classifier.weight/bias 新初始化属于正常现象，需要微调学习。

## 数据与分词器

- train.txt 180,000 条，dev.txt、test.txt 各 10,000 条。
- 每行 UTF-8 `标题<Tab>整数标签`，编号对应 class.txt 的行号。
- pretrained/bert-base-chinese 提供 config.json、tokenizer.json、tokenizer_config.json 和 vocab.txt。
- data.py 只使用 AutoTokenizer，不需要大型权重；train.py 需要本地 PyTorch 预训练权重。
- 训练前需准备与 bert-base-chinese 配置匹配的 PyTorch 权重，放在同一预训练目录。
  本地 .bin、.h5、.msgpack 权重不上传；只用 PyTorch 时无需准备全部框架权重。

## 函数与数据流

| 函数 / 类 | 输入 | 处理与输出 |
| --- | --- | --- |
| Config.__init__ | 无 | 路径、batch_size、max_length、训练参数、设备配置 |
| load_class_list | 配置中的类别文件 | 按编号排列的类别列表 |
| load_news_data(file_path, max_samples) | 数据路径、可选样本上限 | 跳过空行，按 Tab 拆分，返回 (text, int(label)) 列表，格式异常报告行号 |
| NewDataSet.__init__ | samples | 保存样本 |
| NewDataSet.__len__ | 无 | 样本数量 |
| NewDataSet.__getitem__ | 索引 | 一条 (text, label) |
| build_context_fn | tokenizer、可选 max_length | 返回 DataLoader 使用的 collate_fn |
| 内部 collate_fn | 一批样本 | 分词、动态 padding、截断、添加 long 类型 labels |
| build_dataloader | 路径、tokenizer、shuffle，可选 max_samples/config | 读取数据，按指定配置构建 Dataset 和 DataLoader |
| metrics.py / compute_metrics | 真实/预测标签、类别总数 | accuracy 和完整类别集合上的宏平均 P/R/F1，零除记为 0 |
| train.py / set_seed | seed | 固定 Python、NumPy、PyTorch 随机源 |
| build_optimizer | model、config | 对矩阵权重衰减，对 bias/LayerNorm 等一维参数不衰减 |
| train_one_epoch | 模型、loader、优化器、调度器、配置、轮次 | 更新参数，返回按样本加权的平均 loss 和准确率 |
| evaluate | 模型、loader、设备、类别数 | 无梯度验证，返回 dev loss 与分类指标 |
| save_best_model | 模型、tokenizer、路径、dev 指标、轮次 | 保存可重新加载的模型、分词器、最佳指标 |
| main | 可选命令行参数 | 组织加载、训练、验证、保存、记录与早停 |

数据流：TXT → samples → NewDataSet → DataLoader → collate_fn → 模型输入字典。
批次包含 input_ids、attention_mask、通常还有 token_type_ids，以及 labels。
默认 batch_size=16、max_length=64、num_workers=0，按 CUDA 是否可用选择设备。
动态 padding 仅补到当前批次最长文本，因此长度不一定是 64。

已读取 180,000 条训练样本，首个非随机 batch 的 input_ids=(16,26)，labels=(16,)。
训练入口已使用本地真实预训练权重，在临时目录完成 2 条样本的一次参数更新、
dev 验证、最佳模型保存及重新加载检查，没有启动全量训练或覆盖已有模型。

## 今日修复

1. load_news_data 定义与调用不一致：接收 file_path，并用它代替不存在的 conf.news_path。
2. conf.max_len 不存在：统一为 conf.max_length。
3. 分词器路径从 ../pretrained 改为 conf.pretrained_model_path，避免工作目录影响。
