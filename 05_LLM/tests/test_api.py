"""HTTP 输入、输出与错误处理检查，不调用真实服务。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import api
from deepseek_client import DeepSeekClientError


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def test_health_does_not_classify(self):
        with patch.object(api, 'classify_news') as classify:
            self.assertEqual(self.client.get('/health').json, {'status': 'OK'})
            classify.assert_not_called()

    def test_valid_response_maps_class_name(self):
        with patch.object(api, 'classify_news', return_value={
                'class_name': 'sports', 'reason': '体育比赛', 'total_usage': 12}) as classify:
            response = self.client.post('/predict', json={'text': '  球队夺冠  '})
            classify.assert_called_once_with('球队夺冠')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['data']['category'], 'sports')

    def test_invalid_inputs_never_call_api(self):
        with patch.object(api, 'classify_news') as classify:
            for data in ([], {}, {'text': None}, {'text': 123}, {'text': ''},
                         {'text': '   '}, {'text': '字' * 5001}):
                with self.subTest(data_type=type(data).__name__):
                    self.assertEqual(self.client.post('/predict', json=data).status_code, 400)
            self.assertEqual(self.client.post('/predict', data='{', content_type='application/json').status_code, 400)
            classify.assert_not_called()

    def test_service_and_internal_errors(self):
        with patch.object(api, 'classify_news', side_effect=DeepSeekClientError('模拟连接失败')):
            self.assertEqual(self.client.post('/predict', json={'text': '新闻'}).status_code, 502)
        with patch.object(api, 'classify_news', side_effect=RuntimeError('private internal detail')):
            response = self.client.post('/predict', json={'text': '新闻'})
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('private', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
