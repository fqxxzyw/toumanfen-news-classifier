# BERT INT8 动态量化

在项目根目录使用 `toumanfen` 环境执行：

```powershell
python 06_model_quantization/src/quantize_model.py
```

FP32 模型来自 `models/bert_fp32/best_model`，该目录必须直接包含 `config.json`、`model.safetensors` 和分词器文件。脚本使用本地微调权重，在 CPU 上量化 Linear 层，并先执行一次推理检查。原始 FP32 文件保留。

结果保存在 `models/bert_int8/`：`model_state.pt` 是量化后的 state_dict，同目录还包含重建模型所需的配置与分词器。它不是普通的 Hugging Face 模型权重文件，应使用专门的加载函数。在本阶段的 `src` 目录下编写推理脚本时可这样使用：

```python
import torch
from transformers import AutoTokenizer
from config import config
from quantize_model import load_int8_model

model = load_int8_model()
tokenizer = AutoTokenizer.from_pretrained(
    config.int8_state_path.parent, local_files_only=True,
)
inputs = tokenizer('中国队在决赛中获得冠军。', return_tensors='pt',
                   truncation=True, max_length=config.max_length)
with torch.inference_mode():
    logits = model(**inputs).logits
predicted_id = logits.argmax(dim=-1).item()
print(model.config.id2label[predicted_id])
```

保存的是 TorchAO 张量，需要兼容的 PyTorch、TorchAO 和 Transformers 环境。当前验证环境为 PyTorch 2.14.1+cu126、TorchAO 0.18.0、Transformers 5.19.0，推理设备是 CPU。Embedding 和 LayerNorm 等未量化，模型文件不会严格缩小为原来的四分之一。

## 比较分类质量

在项目根目录运行以下命令，先用 64 条样本验证加载、分词与评估流程：

```powershell
python 06_model_quantization/src/evaluate.py --max-samples 64 --batch-size 8 --threads 4
```

去掉 `--max-samples` 即评估整个测试集；在 PyCharm 直接运行 `evaluate.py` 也默认评估全部数据。`--batch-size` 和 `--threads` 可选，不指定时采用项目批大小与 PyTorch 默认线程设置。

脚本使用同一个分词器、相同样本顺序与截断长度，先评估 FP32，再加载已有 INT8 权重评估，不会重新量化。输出 Accuracy、宏平均 Precision、Recall、F1 和各类别报告。宏平均覆盖模型定义的全部类别，零分母按 0 处理。`--max-samples` 只读取文件开头的 N 条，若文件按类别排列，小样本的宏平均不能代表完整测试集，正式结论应来自全量评估。

本脚本比较分类质量；推理速度需要另做预热和计时实验。

## 比较权重大小和推理延迟

在仓库根目录运行：

```powershell
python 06_model_quantization/src/model_compare.py
python 06_model_quantization/src/benchmark.py --batch-size 1 --warmup 1 --repeat 3 --threads 4
```

大小比较读取 FP32 safetensors 与 INT8 state_dict 文件；按 1024² 字节换算，实际单位为 MiB。当前本地文件约为 390.17 MiB 与 145.96 MiB，压缩倍数约 2.67。

延迟比较固定同一批次和 CPU 线程数，分词、模型加载、标签损失计算不计时。先预热，再报告每批延迟的 p50 和 nearest-rank p95，单位 ms。上面的 3 次计时仅用于流程验证；正式测量应提高预热和重复次数，并记录输入形状、线程数和环境。压缩文件不等于保证推理加速。

运行小数据回归检查：

```powershell
python -m unittest discover -s 06_model_quantization/tests -p 'test_*.py' -v
```
