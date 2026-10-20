"""离线检查：不使用真实密钥，也不请求外部 API。"""

import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / 'src'))
import deepseek_client as module


def response(content, finish_reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=content)
    )])


class ClientTests(unittest.TestCase):
    def client(self, result):
        client = module.DeepSeekClient(api_key='offline-test-key')
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = result
        return client

    def test_valid_result_and_payload(self):
        client = self.client(response('{"category":"sports","reason":"体育比赛结果"}'))
        result = client.classify('球队夺冠')
        self.assertEqual(result['category'], 'sports')
        arguments = client._client.chat.completions.create.call_args.kwargs
        self.assertEqual(arguments['response_format'], {'type': 'json_object'})
        self.assertEqual(arguments['extra_body']['thinking']['type'], 'disabled')
        self.assertIn('球队夺冠', arguments['messages'][1]['content'])

    def test_invalid_output_is_rejected(self):
        for content in ('not json', '[]', '{"category":"unknown","reason":"理由"}',
                        '{"category":[],"reason":"理由"}', '{"category":"sports","reason":""}'):
            with self.subTest(content=content):
                with self.assertRaises(module.DeepSeekClientError):
                    self.client(response(content)).classify('球队夺冠')

    def test_input_and_truncation(self):
        client = self.client(response('{}', 'length'))
        for title in ('', '   ', None, 123, 'x' * 5001):
            with self.assertRaises(module.DeepSeekClientError):
                client.classify(title)
        client._client.chat.completions.create.assert_not_called()
        with self.assertRaisesRegex(module.DeepSeekClientError, '长度上限'):
            client.classify('球队夺冠')

    def test_cli_single_and_interactive(self):
        fake = MagicMock()
        fake.configured = True
        fake.classify.return_value = {'category': 'sports', 'reason': '体育比赛结果'}
        with patch.object(module, 'DeepSeekClient', return_value=fake), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(module.main(['--text', '球队夺冠']), 0)
        self.assertEqual(json.loads(output.getvalue())['category'], 'sports')
        fake.classify.reset_mock()
        with patch.object(module, 'DeepSeekClient', return_value=fake), patch('builtins.input', side_effect=['球队夺冠', 'q']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(module.main([]), 0)
        fake.classify.assert_called_once_with('球队夺冠')

    def test_missing_key_does_not_call_api(self):
        with patch.object(module, 'read_local_settings', return_value={}):
            client = module.DeepSeekClient()
        with patch.object(module, 'OpenAI') as sdk:
            with self.assertRaisesRegex(module.DeepSeekClientError, 'DEEPSEEK_API_KEY'):
                client.classify('球队夺冠')
            sdk.assert_not_called()


class PageTests(unittest.TestCase):
    def test_page_prediction_and_empty_input(self):
        fake = MagicMock()
        fake.classify.return_value = {'category': 'sports', 'reason': '新闻涉及体育比赛结果'}
        with patch.object(module, 'read_local_settings', return_value={'DEEPSEEK_API_KEY': 'offline-test-key'}), patch.object(module, 'DeepSeekClient', return_value=fake):
            page = AppTest.from_file(str(PROJECT_DIR / 'src/app.py'), default_timeout=15).run()
            self.assertFalse(page.exception)
            next(button for button in page.button if button.label == '🏀 体育赛事').click().run()
            self.assertTrue(page.text_area[0].value)
            next(button for button in page.button if button.label == '🐋 开始分类').click().run()
            self.assertFalse(page.exception)
            self.assertEqual(page.metric[0].value, '体育')
            self.assertEqual(len(page.session_state['records']), 1)
            self.assertEqual(fake.classify.call_count, 1)
            fake.close.assert_called_once()
            page.text_area[0].set_value(' ')
            next(button for button in page.button if button.label == '🐋 开始分类').click().run()
            self.assertTrue(page.warning)
            self.assertFalse(page.metric)
            self.assertEqual(fake.classify.call_count, 1)


if __name__ == '__main__':
    unittest.main()
