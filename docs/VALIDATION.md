# 验证记录

## 2026-10-21 · LLM 评估与 INT8 实验

环境：Windows、Python 3.10.22、Anaconda toumanfen、PyTorch 2.14.1+cu126、TorchAO 0.18.0、Transformers 5.19.0。CPU 推理；没有重新训练 BERT。

| 检查 | 结果与边界 |
| --- | --- |
| LLM 自动检查 | 23 项通过；HTTP、批处理、分类封装、用量、CSV 标签兼容与指标；外部调用全部模拟 |
| 量化自动检查 | 10 项通过；数据与标签校验、完整样本汇总、权重加载入口、计时与 p95 |
| 实际量化 | 74 个 Linear 层转为 TorchAO INT8；保存配置、分词器和 state_dict |
| 实际恢复 | 新进程恢复，3 条测试新闻批量推理、有限 logits、重复推理一致性通过 |
| 实际质量评估 | 同一测试集前 64 条，均为 education；FP32/INT8 Accuracy 均 0.953125，十类宏平均 F1 均约 0.0976；不代表全量成绩 |
| 实际权重大小 | FP32 390.1718 MiB、INT8 145.9637 MiB，压缩 2.6731 倍 |
| 延迟入口试跑 | 同一批 `(1,22)`、4 线程、预热 1 次、计时 3 次；FP32 p50/p95 98.025/116.466 ms，INT8 2318.776/2367.840 ms；仅作流程检查，不是稳定基准 |

从仓库根目录复现自动检查：

```powershell
python -m unittest discover -s 05_LLM/tests -p 'test_*.py' -v
python -m unittest discover -s 06_model_quantization/tests -p 'test_*.py' -v
```

真实模型验证需自行准备权重，命令及限制见 [今日进度](progress/2026-10-21.md) 和 [量化教程](../06_model_quantization/README.md)。本次发布整理不调用真实 DeepSeek API，也不上传本地模型或预测产物。

## 2026-10-20 · BERT 与 DeepSeek 开发更新

环境：Windows、Anaconda `toumanfen`、Python 3.10.22。验证不覆盖现有训练权重，不报告完整 test 成绩。

| 范围 | 检查内容 | 结果 |
| --- | --- | --- |
| BERT API / 页面 | unittest 的 6 项检查：请求格式、合法预测、非法输入、错误请求、示例填充、空文本、服务断连与超时 | 通过 |
| BERT 本地权重 | 模型与 tokenizer 从 best_model 加载，不访问原始预训练目录 | 通过，无 missing/unexpected/mismatched 权重 |
| BERT 训练流程 | 小模型执行参数更新、验证、最优模型元数据保存、连续未提升早停 | 通过，未重新运行完整 BERT 训练 |
| BERT 测试入口 | 使用本地真实模型和 test.txt 的前 16 条样本运行 main | 通过，不构成全量 test 指标 |
| BERT 单条预测 | 验证输出类别 ID 和分数等于 softmax 概率最大值对应项 | 通过 |
| BERT 页面到接口 | Streamlit AppTest 提交标题，通过真实 HTTP 调用本地 Flask，再使用已保存的 BERT 推理 | 通过，页面显示类别和分数 |
| DeepSeek 命令行 / 页面 | unittest 的 6 项离线检查：请求格式、JSON/类别校验、输入/截断处理、单条与交互命令行、缺少密钥、页面提交 | 通过，全部使用模拟响应 |
| DeepSeek SDK | 已安装 openai 3.26.0、python-dotenv 1.2.2、tenacity 9.1.4，验证 SDK 初始化和 --help 入口 | 通过，没有发送外部请求 |
| Streamlit 服务 | 本地 8501、8502 页面的健康端点 | 启动时均返回 ok；后台服务不是仓库内容 |

复现离线检查时，分别进入对应目录运行：

```powershell
# 04_bert
python -m unittest discover -s tests -p test_web.py -v

# 05_LLM
python -m unittest discover -s tests -p test_deepseek.py -v
```

DeepSeek 真实调用需用户在本地设置有效密钥与账户额度。`.env` 和密钥不进入源码提交，仓库仅包含 `.env.example`。
本次不重新训练完整 BERT、不重新评估全量 test，也未进行负载或生产部署验证。

## v0.2.0 历史验证记录

日期：2026-10-18。环境：Windows、Anaconda toumanfen、Python 3.10.22。

## 环境版本

当前安装元数据：fasttext-wheel 0.9.2、NumPy 2.2.5、scikit-learn 1.7.2、
jieba 0.42.1、Flask 3.1.3、Streamlit 1.64.0、requests 2.34.2、
torch 2.14.1、transformers 5.19.0。
这记录已有环境，不代表已经在干净环境重建全部依赖；GPU 版本应按设备选择。

## 实际执行的检查

| 范围 | 内容 | 结果 |
| --- | --- | --- |
| 语法 | 03_fasttext、04_bert 的 9 个 Python 文件 AST 解析 | 通过 |
| 预处理 | 临时样本的 char/word、标签与重复运行覆盖 | 通过 |
| 基础训练/保存 | 临时小样本、低维参数，调用两个训练函数并保存真实模型 | 通过，不覆盖已有模型 |
| 中文路径 | 相同样本的中文绝对、英文相对和英文绝对路径训练/保存 | 当前 fastText 中文绝对路径失败，后两种通过 |
| 调参传参 | 两模式经过实际 Python 参数解析，在原生训练入口前截停 | 参数名、300 秒预算、训练/验证路径检查通过；未重新执行搜索 |
| 四模型评估 | 本地四个已有模型各评估对应 dev 10,000 条 | 均成功，见下表 |
| 单条预测 | 字符调参模型、char 预处理、批量预测接口 | 返回类别和概率 |
| API | GET http://127.0.0.1:8000/health | status=OK |
| Streamlit | 模块启动，GET http://127.0.0.1:8501/_stcore/health | ok；不是完整浏览器交互测试 |
| 发布复查 | AppTest 点击预测，requests.post 注入真实 Flask test_client，加载已有字符调参模型 | /health、/predict、结果中的 label/probability 和页面 JSON 显示通过；不依赖存活的后台服务 |
| BERT | 本地 tokenizer、读取 train、取首个非随机 batch | 180,000 条，input_ids=(16,26)，labels=(16,) |

## FastText dev 指标

| 模型 | 准确率 | 样本数 |
| --- | ---: | ---: |
| fasttext_char | 0.8717 | 10,000 |
| fasttext_word | 0.8990 | 10,000 |
| fasttext_char_autotune | 0.9143 | 10,000 |
| fasttext_word_autotune | 0.9155 | 10,000 |

dev 也用于自动调参，因此不属于最终独立 test 成绩。
发布整理使用已有模型，没有重新全量训练或覆盖模型；其他环境重训结果可能不同。

## 当前限制

- 相对路径方案要求工作目录位于项目内，使交给 fastText 的路径不含中文。
- 当前 requests 报过编码检测依赖警告，这与 Streamlit 启动方式不同。
- BERT 只验证数据输入，未训练，不报告分类准确率。
- API 异常输入校验、网页请求超时、完整浏览器交互、负载和生产部署未完善。
- 未执行最终 test 基准、量化、蒸馏或资源优化。
