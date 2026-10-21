"""验证批处理使用 classify_news 的真实返回字段，不请求 API。"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / 'src'))
spec = importlib.util.spec_from_file_location('batch_fields', PROJECT_DIR / 'src/prodict-one.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchFieldTests(unittest.TestCase):
    def test_wrapper_result_is_saved_as_success_without_optional_token_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'predictions.csv'
            sample = pd.DataFrame([dict(row_id=0, news='球队夺冠', y_true=7)])
            # 只模拟抽样得到的一条记录，避免实际读取开发集或执行 500 条预测。
            with (patch.object(batch, 'output_dir', output),
                  patch.object(batch, 'load_data', return_value=sample),
                  patch.object(pd.DataFrame, 'sample', return_value=sample),
                  patch.object(batch, 'classify_news', return_value={
                      'class_name': 'sports', 'reason': '体育比赛', 'total_usage': 12,
                  }) as classify,
                  contextlib.redirect_stdout(io.StringIO())):
                batch.run_batch()
                batch.run_batch()  # 第二次应该跳过已经成功的记录。
            classify.assert_called_once_with('球队夺冠')
            rows = pd.read_csv(output, keep_default_na=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.iloc[0]['status'], 'success')
            self.assertEqual(rows.iloc[0]['pred_label'], 'sports')
            self.assertEqual(rows.iloc[0]['prompt_tokens'], '')
            self.assertEqual(rows.iloc[0]['completion_tokens'], '')


if __name__ == '__main__':
    unittest.main()
