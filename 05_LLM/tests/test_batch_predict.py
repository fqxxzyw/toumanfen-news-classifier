"""按当前批处理入口检查成功跳过、失败重试与 CSV 追加，不调用真实 API。"""

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
from deepseek_client import DeepSeekClientError

spec = importlib.util.spec_from_file_location('batch_predict', PROJECT_DIR / 'src/prodict-one.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def prediction(news):
    return {'class_name': 'sports', 'reason': '模拟分类结果', 'total_usage': 12}


class BatchPredictTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'dev.txt'
        self.output = Path(self.temp.name) / 'predictions.csv'
        self.source.write_text('球队夺冠\t7\n芯片发布\t4\n123\t5\n', encoding='utf-8')
        self.output_patch = patch.object(batch, 'output_dir', self.output)
        self.input_patch = patch.object(batch.config, 'dev_path', self.source)
        self.output_patch.start()
        self.input_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.input_patch.stop)

    def run_batch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            batch.run_batch()

    def test_successful_rows_skip_and_write_header_once(self):
        with patch.object(batch, 'classify_news', side_effect=prediction) as classify:
            self.run_batch()
            self.run_batch()
        self.assertEqual(classify.call_count, 3)
        rows = pd.read_csv(self.output)
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(rows['status']), {'success'})
        self.assertEqual(rows.set_index('row_id')['true_lable'].to_dict(), {0: 7, 1: 4, 2: 5})
        self.assertEqual(self.output.read_text(encoding='utf-8-sig').count('row_id,title,'), 1)

    def test_failed_rows_are_retried_and_only_successes_skip(self):
        def sometimes_fails(news):
            if news == '芯片发布':
                raise DeepSeekClientError('模拟连接失败')
            return prediction(news)
        with patch.object(batch, 'classify_news', side_effect=sometimes_fails):
            self.run_batch()
        self.assertEqual(batch.load_finish_ids(), {0, 2})
        rows = pd.read_csv(self.output, keep_default_na=False)
        failed = rows.loc[rows['status'] == 'fail'].iloc[0]
        self.assertEqual(failed['pred_label'], '')
        self.assertEqual(failed['error'], '模拟连接失败')
        with patch.object(batch, 'classify_news', side_effect=prediction) as classify:
            self.run_batch()
        classify.assert_called_once_with('芯片发布')
        self.assertEqual(batch.load_finish_ids(), {0, 1, 2})
        self.assertEqual(len(pd.read_csv(self.output)), 4)

    def test_append_to_empty_file_adds_header(self):
        self.output.touch()
        batch.append_result({'row_id': 0, 'status': 'success'})
        self.assertEqual(pd.read_csv(self.output).iloc[0]['status'], 'success')


if __name__ == '__main__':
    unittest.main()
