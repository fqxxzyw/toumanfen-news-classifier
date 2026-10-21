"""小数据回归检查，不下载模型、不重新量化，也不改写本地权重。"""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dataset
import evaluate


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        self.options = kwargs
        return {'input_ids': torch.ones(len(texts), 2, dtype=torch.long),
                'attention_mask': torch.ones(len(texts), 2, dtype=torch.long)}


class PredictionModel(torch.nn.Module):
    """输入第一个数字就是预测类别，用来手算并验证评估指标。"""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_labels=3, id2label={0: 'a', 1: 'b', 2: 'c'})

    def forward(self, input_ids):
        assert not self.training
        assert not torch.is_grad_enabled()
        logits = torch.nn.functional.one_hot(input_ids[:, 0], num_classes=3).float()
        return SimpleNamespace(logits=logits)


class DatasetEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'test.txt'

    def write(self, text):
        self.path.write_text(text, encoding='utf-8-sig')
        return self.path

    def test_read_bom_blank_lines_limit_and_embedded_tab(self):
        self.write('\n球队\t夺冠\t7\n\n股票上涨\t2\n错误行\n')
        self.assertEqual(dataset.load_data(self.path, max_samples=2),
                         [('球队\t夺冠', 7), ('股票上涨', 2)])

    def test_invalid_rows_have_line_numbers(self):
        for bad in ('无分隔符', '\t7', '新闻\t', '新闻\t文字', '新闻\t1.5', '新闻\t-1'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, '第 2 行'):
                dataset.load_data(self.write('\n' + bad))

    def test_loader_supports_string_device_and_keeps_last_batch(self):
        self.write('新闻甲\t0\n新闻乙\t1\n新闻丙\t2\n')
        tokenizer = StubTokenizer()
        for device in ('cpu', torch.device('cpu')):
            # 不提供 num_workers，模拟本项目当前 Config 的结构。
            config = SimpleNamespace(batch_size=2, max_length=8, device=device)
            loader = dataset.build_dataloader(self.path, tokenizer, config=config)
            batches = list(loader)
            self.assertEqual([len(batch['labels']) for batch in batches], [2, 1])
            self.assertEqual(torch.cat([batch['labels'] for batch in batches]).tolist(), [0, 1, 2])
            self.assertEqual(batches[0]['labels'].dtype, torch.long)
            self.assertFalse(loader.pin_memory)
        self.assertEqual(tokenizer.options['max_length'], 8)
        self.assertTrue(tokenizer.options['truncation'])

    def test_empty_data_and_invalid_sample_limits(self):
        self.write('\n')
        with self.assertRaises(ValueError):
            dataset.build_dataloader(self.path, StubTokenizer())
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dataset.load_data(self.path, max_samples=value)

    def test_metrics_accumulate_samples_instead_of_averaging_batches(self):
        batches = [
            {'input_ids': torch.tensor([[0], [0]]), 'labels': torch.tensor([0, 1])},
            {'input_ids': torch.tensor([[2]]), 'labels': torch.tensor([2])},
        ]
        result = evaluate.evaluate_model(PredictionModel(), batches)
        self.assertEqual(result['samples'], 3)
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['precision'], 0.5)
        self.assertAlmostEqual(result['recall'], 2 / 3)
        self.assertAlmostEqual(result['f1'], 5 / 9)
        self.assertEqual(batches[0]['labels'].tolist(), [0, 1])  # 评估不删除原始 labels。

    def test_invalid_evaluation_labels_and_empty_loader(self):
        with self.assertRaises(ValueError):
            evaluate.evaluate_model(PredictionModel(), [])
        for labels in (torch.tensor([3]), torch.tensor([-1]), torch.tensor([0.0]), torch.tensor([[0]])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate.evaluate_model(PredictionModel(),
                                        [{'input_ids': torch.tensor([[0]]), 'labels': labels}])

    def test_nonfinite_or_wrong_shape_output_is_rejected(self):
        model = PredictionModel()
        batch = [{'input_ids': torch.tensor([[0]]), 'labels': torch.tensor([0])}]
        for logits in (torch.full((1, 3), float('nan')), torch.ones(1, 2)):
            with patch.object(model, 'forward', return_value=SimpleNamespace(logits=logits)):
                with self.assertRaises(ValueError):
                    evaluate.evaluate_model(model, batch)

    def test_int8_loader_reuses_saved_checkpoint_loader(self):
        checkpoint = Path(self.temp.name) / 'model_state.pt'
        checkpoint.touch()
        with (patch.object(evaluate.config, 'int8_state_path', checkpoint),
              patch.object(evaluate, 'load_saved_int8_model') as restore):
            self.assertIs(evaluate.load_int8_model(), restore.return_value)
            restore.assert_called_once_with(checkpoint)


if __name__ == '__main__':
    unittest.main()
