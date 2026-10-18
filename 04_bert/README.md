# BERT 数据加载阶段

当前实现配置、数据读取、Dataset、批次分词和 DataLoader。
**尚未实现分类模型构建、微调训练、验证指标或预测接口。**

## 运行

从本目录执行：

```bat
conda activate toumanfen
python -m pip install -r requirements.txt
python src/data.py
```

入口打印类别并创建 DataLoader，不启动训练。配置按脚本位置定位数据和本地分词器。

## 数据与分词器

- train.txt 180,000 条，dev.txt、test.txt 各 10,000 条。
- 每行 UTF-8 `标题<Tab>整数标签`，编号对应 class.txt 的行号。
- pretrained/bert-base-chinese 提供 config.json、tokenizer.json、tokenizer_config.json 和 vocab.txt。
- 当前只使用 AutoTokenizer，数据加载不需要大型模型权重。
- 后续微调需自行准备与 bert-base-chinese 配置匹配的 PyTorch 权重，放在同一预训练目录。
  本地 .bin、.h5、.msgpack 权重不上传；只用 PyTorch 时无需准备全部框架权重。

## 函数与数据流

| 函数 / 类 | 输入 | 处理与输出 |
| --- | --- | --- |
| Config.__init__ | 无 | 路径、batch_size、max_length、训练参数、设备配置 |
| load_class_list | 配置中的类别文件 | 按编号排列的类别列表 |
| load_news_data(file_path) | 数据路径 | 跳过空行，按 Tab 拆分，返回 (text, int(label)) 列表，格式异常报告行号 |
| NewDataSet.__init__ | samples | 保存样本 |
| NewDataSet.__len__ | 无 | 样本数量 |
| NewDataSet.__getitem__ | 索引 | 一条 (text, label) |
| build_context_fn | tokenizer | 返回 DataLoader 使用的 collate_fn |
| 内部 collate_fn | 一批样本 | 分词、动态 padding、截断、添加 long 类型 labels |
| build_dataloader | file_path、tokenizer、shuffle | 读取数据，构建 Dataset 和 DataLoader |

数据流：TXT → samples → NewDataSet → DataLoader → collate_fn → 模型输入字典。
批次包含 input_ids、attention_mask、通常还有 token_type_ids，以及 labels。
默认 batch_size=16、max_length=64、num_workers=0，按 CUDA 是否可用选择设备。
动态 padding 仅补到当前批次最长文本，因此长度不一定是 64。

已读取 180,000 条训练样本，首个非随机 batch 的 input_ids=(16,26)，labels=(16,)。
本次未加载 BERT 模型权重、执行微调或测试 GPU 性能。

## 今日修复

1. load_news_data 定义与调用不一致：接收 file_path，并用它代替不存在的 conf.news_path。
2. conf.max_len 不存在：统一为 conf.max_length。
3. 分词器路径从 ../pretrained 改为 conf.pretrained_model_path，避免工作目录影响。
