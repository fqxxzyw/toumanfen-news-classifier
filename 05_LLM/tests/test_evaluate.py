"""评估逻辑回归测试：仅使用临时 CSV，不访问 API。"""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evalute import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.classes = self.root / 'class.txt'
        # 特意改变类别顺序，验证 ID 取自 class.txt，而不是 prompts 字典。
        self.classes.write_text('sports\nfinance\n', encoding='utf-8')
        self.csv_path = self.root / 'predictions.csv'

    def evaluate(self, rows):
        pd.DataFrame(rows).to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        return evaluate_predictions(self.csv_path, self.classes)

    def test_old_typo_numeric_labels(self):
        result = self.evaluate([
            dict(row_id=0, status='success', true_lable=0, pred_label='sports'),
            dict(row_id=1, status='success', true_lable=1, pred_label='sports'),
        ])
        self.assertEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['precision'], 0.25)
        self.assertAlmostEqual(result['recall'], 0.5)
        self.assertAlmostEqual(result['f1'], 1 / 3)

    def test_new_schema_deduplicates_latest_success(self):
        result = self.evaluate([
            dict(row_id=0, status='failed', true_id=0, true_label='sports', pred_label=''),
            dict(row_id=0, status='success', true_id=0, true_label='sports', pred_label='finance'),
            dict(row_id=0, status='success', true_id=0, true_label='sports', pred_label='sports'),
            dict(row_id=0, status='fail', true_id=0, true_label='sports', pred_label=''),
            dict(row_id=1, status='success', true_id=1, true_label='finance', pred_label='finance'),
            dict(row_id=2, status='fail', true_id=1, true_label='finance', pred_label=''),
        ])
        self.assertEqual((result['total'], result['evaluated'], result['unsuccessful']), (3, 2, 1))
        self.assertAlmostEqual(result['success_rate'], 2 / 3)
        self.assertEqual(result['f1'], 1.0)

    def test_true_label_names_only(self):
        result = self.evaluate([dict(row_id=0, status='success', true_label='sports', pred_label='sports')])
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['f1'], 0.5)  # 两类宏平均，包括未出现的 finance 类。

    def test_invalid_labels_and_conflicting_true_columns(self):
        for values in (
            dict(true_label='unknown', pred_label='sports'),
            dict(true_label='sports', pred_label=''),
            dict(true_label=0.5, pred_label='sports'),
            dict(true_id=0, true_label='finance', pred_label='sports'),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.evaluate([dict(row_id=0, status='success', **values)])

    def test_no_success_or_missing_schema(self):
        for row in (
            dict(row_id=0, status='fail', true_lable=0, pred_label=''),
            dict(row_id=0, status='success', pred_label='sports'),
            dict(status='success', true_lable=0, pred_label='sports'),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.evaluate([row])


if __name__ == '__main__':
    unittest.main()
