"""将已微调的 FP32 BERT 量化成 INT8，并提供对应的离线加载方法。"""

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

from config import config


def quantize_model(fp32_model_dir=None, int8_state_path=None):
    """从本地加载已训练模型，动态量化 Linear 层，在 CPU 验证后保存。"""
    source = Path(config.fp32_model_dir if fp32_model_dir is None else fp32_model_dir)
    destination = Path(config.int8_state_path if int8_state_path is None else int8_state_path)

    # Transformers 要求 config.json 就在传入目录的第一层，不会递归查找子目录。
    # 提前给出明确错误，避免把目录问题误认为模型架构不受支持。
    config_path = source / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'模型目录中没有 config.json：{source}；请检查是否还套了一层 best_model。')
    model_config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not model_config.get('model_type'):
        raise ValueError(f'{config_path} 缺少 model_type，请使用完整的已训练模型目录。')
    if destination.parent.resolve() == source.resolve():
        raise ValueError('INT8 输出目录应与 FP32 模型目录分开，避免覆盖原始配置。')

    # local_files_only=True 保证使用本地微调模型，不会下载同名预训练模型。
    # 本实验在 CPU 上运行，量化前统一为 float32，并用 eval 关闭 Dropout。
    print(f'加载 FP32 模型：{source}', flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True)
    model = model.cpu().float().eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)

    # quantize_ 是原地修改：Linear 权重转为 INT8，激活在推理时动态量化。
    # Embedding、LayerNorm 等仍保留浮点数，所以整个文件不会恰好缩小到 1/4。
    # 不 deepcopy 整个 BERT，可避免额外复制约 390 MB 的 FP32 参数。
    print('开始 INT8 动态量化……', flush=True)
    quantize_(model, Int8DynamicActivationInt8WeightConfig())

    # 保存之前真正执行一次 CPU 前向计算，确认当前环境支持量化后的运算。
    inputs = tokenizer('中国队在决赛中获得冠军。', return_tensors='pt',
                       truncation=True, max_length=config.max_length)
    with torch.inference_mode():
        logits = model(**inputs).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError('量化后推理出现非有限值，未保存模型。')

    destination.parent.mkdir(parents=True, exist_ok=True)
    # 配置和分词器用于之后离线重建模型，实际量化参数保存在 model_state.pt。
    # 这不是普通的 Hugging Face safetensors 目录，加载时使用下面的 load_int8_model。
    model.config.save_pretrained(destination.parent)
    tokenizer.save_pretrained(destination.parent)
    torch.save(model.state_dict(), destination)

    # 仅比较模型权重文件大小，分词器和配置不包含在这个统计中。
    source_weights = list(source.glob('*.safetensors')) + list(source.glob('pytorch_model*.bin'))
    source_size = sum(path.stat().st_size for path in source_weights)
    target_size = destination.stat().st_size
    print(f'INT8 权重已保存：{destination}', flush=True)
    if source_size:
        print(f'权重大小：FP32 {source_size / 1024 ** 2:.2f} MiB → INT8 {target_size / 1024 ** 2:.2f} MiB')
    print('CPU 前向推理检查通过。')
    return model


def load_int8_model(int8_state_path=None):
    """从配置和 INT8 state_dict 恢复模型，无需再次加载 FP32 权重或重新量化。"""
    path = Path(config.int8_state_path if int8_state_path is None else int8_state_path)
    model_config = AutoConfig.from_pretrained(path.parent, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(model_config).cpu().float()
    # 导入 torchao 后，当前版本的 INT8 张量类型已注册，支持 weights_only=True。
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    # assign=True 保留 state_dict 中的 INT8 张量类型，不能复制进普通浮点参数。
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


# 只有直接运行本文件才执行量化；其他脚本可安全导入 load_int8_model。
if __name__ == '__main__':
    quantize_model()
