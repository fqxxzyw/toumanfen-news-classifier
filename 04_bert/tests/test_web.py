"""运行：python -m unittest discover -s tests -p test_web.py。无需加载真实模型。"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / 'src'))
from api import create_app


class FakePredictor:
    device = 'cpu'
    max_len = 64
    class_list = ['sports']

    def __init__(self):
        self.calls = []

    def predict_one(self, text):
        self.calls.append(text)
        return {'pred': 0, 'class': 'sports', 'score': 0.95}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.predictor = FakePredictor()
        self.client = create_app(self.predictor).test_client()

    def test_prediction_and_health(self):
        self.assertEqual(self.client.get('/health').json['status'], 'ok')
        response = self.client.post('/predict', json={'text': '  球队夺冠  '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'pred': 0, 'class': 'sports', 'score': 0.95})
        self.assertEqual(self.predictor.calls, ['球队夺冠'])

    def test_invalid_input_does_not_call_model(self):
        for payload in ({}, {'text': ''}, {'text': '  '}, {'text': 123},
                        {'text': None}, {'text': []}, [], {'text': 'a' * 5001}):
            with self.subTest(payload=repr(payload)[:60]):
                response = self.client.post('/predict', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.json)
        self.assertEqual(self.predictor.calls, [])

    def test_malformed_large_and_unknown_requests(self):
        self.assertEqual(self.client.post(
            '/predict', data='{', content_type='application/json'
        ).status_code, 400)
        self.assertEqual(self.client.post(
            '/predict', data='a' * 65537, content_type='application/json'
        ).status_code, 413)
        self.assertEqual(self.client.get('/missing').status_code, 404)
        self.assertTrue(self.client.get('/missing').is_json)


def healthy_response(*args, **kwargs):
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {'status': 'ok', 'device': 'cpu', 'max_length': 64,
                      'class_list': ['sports']},
    )


def prediction_response(*args, **kwargs):
    return SimpleNamespace(
        ok=True, status_code=200,
        json=lambda: {'pred': 0, 'class': 'sports', 'score': 0.95},
    )


class PageTests(unittest.TestCase):
    def page(self):
        return AppTest.from_file(str(PROJECT_DIR / 'src/app.py'), default_timeout=15).run()

    def submit(self, page):
        next(button for button in page.button if button.label == '开始预测').click().run()
        self.assertFalse(page.exception)

    @patch('requests.get', side_effect=healthy_response)
    @patch('requests.post', side_effect=prediction_response)
    def test_example_prediction_and_empty_input(self, post, get):
        page = self.page()
        self.assertFalse(page.exception)
        next(button for button in page.button if button.label == '体育新闻').click().run()
        self.assertTrue(page.text_area[0].value)
        self.submit(page)
        self.assertTrue(any(metric.value == '体育' for metric in page.metric))
        self.assertTrue(any(metric.value == '95.00%' for metric in page.metric))
        self.assertEqual(post.call_count, 1)

        page.text_area[0].set_value('   ')
        self.submit(page)
        self.assertTrue(any('请先输入' in item.value for item in page.warning))
        self.assertFalse(page.metric)
        self.assertEqual(post.call_count, 1)

    @patch('requests.get', side_effect=healthy_response)
    @patch('requests.post', side_effect=requests.ConnectionError('offline'))
    def test_offline_service(self, post, get):
        page = self.page()
        page.text_area[0].set_value('测试新闻')
        self.submit(page)
        self.assertTrue(any('无法连接' in item.value for item in page.error))

    @patch('requests.get', side_effect=healthy_response)
    @patch('requests.post', side_effect=requests.Timeout('timeout'))
    def test_timeout(self, post, get):
        page = self.page()
        page.text_area[0].set_value('测试新闻')
        self.submit(page)
        self.assertTrue(any('超时' in item.value for item in page.error))


if __name__ == '__main__':
    unittest.main()
