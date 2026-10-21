"""对固定同一批输入进行 CPU 推理计时，输出每批延迟 p50 与 p95。"""

import argparse
import gc
import math
import statistics
import time

import torch
from transformers import AutoTokenizer

from config import config
from dataset import build_dataloader
from evaluate import load_int8_model, load_fp32


def benchmark_model(model, batch, warmup_steps=20, repeat_step=100):
    """预热后计时；不包含分词、模型加载或标签损失计算的时间。"""
    if warmup_steps < 0 or repeat_step < 1:
        raise ValueError('warmup_steps 不能为负，repeat_step 必须大于 0。')
    model = model.cpu().eval()
    # 在计时前准备输入；labels 仅用于质量评估，不参与推理速度比较。
    inputs = {key: value.cpu() for key, value in batch.items() if key != 'labels'}
    latencies = []
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(**inputs)
        for _ in range(repeat_step):
            start = time.perf_counter()
            model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)
    ordered = sorted(latencies)
    # nearest-rank 百分位：ceil(0.95 * N) 从 1 开始计数，再减 1 转成列表下标。
    return {'p50': statistics.median(latencies),
            'p95': ordered[math.ceil(len(ordered) * 0.95) - 1],
            'latencies': latencies}


def main(argv=None):
    parser = argparse.ArgumentParser(description='FP32 / INT8 固定批次 CPU 延迟比较')
    parser.add_argument('--batch-size', type=int, default=config.batch_size)
    parser.add_argument('--warmup', type=int, default=config.warmup_steps)
    parser.add_argument('--repeat', type=int, default=config.benchmark_steps)
    parser.add_argument('--threads', type=int, default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error('batch-size、repeat 必须为正，warmup 不能为负。')
    if args.threads is not None:
        if args.threads < 1:
            parser.error('threads 必须大于 0。')
        torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(config.fp32_model_dir, local_files_only=True)
    # 仅取一个固定批次，两种模型复用它，保证文本、批大小、补齐长度完全一致。
    from dataclasses import replace
    loader = build_dataloader(config.test_path, tokenizer, shuffle=False,
                              max_samples=args.batch_size,
                              config=replace(config, batch_size=args.batch_size))
    batch = next(iter(loader))
    print(f"CPU 线程数={torch.get_num_threads()}，input_ids={tuple(batch['input_ids'].shape)}，"
          f'预热={args.warmup}，计时次数={args.repeat}', flush=True)
    for name, load_model in [('FP32', load_fp32), ('INT8', load_int8_model)]:
        model = load_model()
        result = benchmark_model(model, batch, args.warmup, args.repeat)
        print(f"{name}：每批 p50={result['p50']:.3f} ms，p95={result['p95']:.3f} ms", flush=True)
        del model
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
