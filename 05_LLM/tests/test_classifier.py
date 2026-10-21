"""分类封装与用量元数据的离线检查，不调用真实 API。"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import classifer
from deepseek_client import DeepSeekClient, DeepSeekClientError


class ClassifierTests(unittest.TestCase):
    def test_passes_title_and_maps_result_with_usage(self):
        client = MagicMock()
        client.classify.return_value = {'category': 'sports', 'reason': '体育比赛结果'}
        client.last_usage = SimpleNamespace(total_tokens=123)
        with patch.object(classifer, 'DeepSeekClient', return_value=client):
            result = classifer.classify_news('  球队夺冠  ')
        client.classify.assert_called_once_with('球队夺冠')
        self.assertEqual(result, {'class_name': 'sports', 'reason': '体育比赛结果', 'total_usage': 123})
        client.close.assert_called_once()

    def test_missing_usage_and_failed_request(self):
        client = MagicMock()
        client.classify.return_value = {'category': 'sports', 'reason': '体育比赛结果'}
        client.last_usage = None
        with patch.object(classifer, 'DeepSeekClient', return_value=client):
            self.assertIsNone(classifer.classify_news('球队夺冠')['total_usage'])
        client.close.assert_called_once()

        client.reset_mock()
        client.classify.side_effect = DeepSeekClientError('连接失败')
        with patch.object(classifer, 'DeepSeekClient', return_value=client):
            with self.assertRaises(DeepSeekClientError):
                classifer.classify_news('球队夺冠')
        client.close.assert_called_once()

    def test_invalid_input_never_constructs_client(self):
        with patch.object(classifer, 'DeepSeekClient') as create:
            for news in ('', '   ', None, [], 123):
                with self.subTest(news=news), self.assertRaises(ValueError):
                    classifer.classify_news(news)
            create.assert_not_called()

    def test_client_captures_response_usage_and_clears_stale_usage(self):
        client = DeepSeekClient(api_key='offline-test-key')
        sdk = MagicMock()
        client._client = sdk
        sdk.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=123),
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content='{"category":"sports","reason":"体育比赛结果"}'
            ))],
        )
        self.assertEqual(client.classify('球队夺冠')['category'], 'sports')
        self.assertEqual(client.last_usage.total_tokens, 123)
        with self.assertRaises(DeepSeekClientError):
            client.classify([])
        self.assertIsNone(client.last_usage)
        client.close()


if __name__ == '__main__':
    unittest.main()
