"""验证计时范围和小样本 p95 计算，不加载大型模型。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from benchmark import benchmark_model


class BenchmarkTests(unittest.TestCase):
    def test_timer_excludes_warmup_and_labels(self):
        model = torch.nn.Linear(1, 1)
        batch = {'input': torch.ones(2, 1), 'labels': torch.tensor([0, 1])}
        with (patch.object(model, 'forward', wraps=model.forward) as forward,
              patch('benchmark.time.perf_counter', side_effect=[0, .001, 1, 1.002, 2, 2.003])):
            result = benchmark_model(model, batch, warmup_steps=2, repeat_step=3)
        self.assertEqual(forward.call_count, 5)
        self.assertNotIn('labels', forward.call_args.kwargs)
        self.assertAlmostEqual(result['p50'], 2)
        self.assertAlmostEqual(result['p95'], 3)
        self.assertEqual(len(result['latencies']), 3)

    def test_invalid_counts(self):
        for warmup, repeat in [(-1, 2), (0, 0)]:
            with self.subTest(warmup=warmup, repeat=repeat), self.assertRaises(ValueError):
                benchmark_model(torch.nn.Linear(1, 1), {'input': torch.ones(1, 1)}, warmup, repeat)


if __name__ == '__main__':
    unittest.main()
