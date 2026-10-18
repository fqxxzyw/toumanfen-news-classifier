---
language: zh
---

# 本地 BERT 中文分词器与配置

版本库保留配置、分词器和词表，供 04_bert/src/data.py 使用。
大型 .bin、.h5、.msgpack 权重只保留在本地，不上传。
数据加载阶段仅需 AutoTokenizer；后续微调应自行准备与配置兼容的 PyTorch 权重。
详细说明见 [BERT 教程](../../README.md)。
