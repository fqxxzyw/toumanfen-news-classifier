"""DeepSeek 新闻十分类客户端：可导入使用，也可直接在命令行交互运行。"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai import InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from config import Config
from prompts import CATEGORY_DESCRIPTIONS, SYSTEM_PROMPT, USER_PROMPT

PROJECT_DIR = Path(__file__).resolve().parents[1]


class DeepSeekClientError(Exception):
    """给命令行和前端展示的可读错误，不包含密钥和请求头。"""


def read_local_settings():
    # 固定从项目目录读取，避免在 PyCharm 与命令行运行时找错 .env。
    # 仅返回配置，不修改进程环境；系统环境变量覆盖本地文件。
    settings = {**dotenv_values(PROJECT_DIR / '.env'),
                **dotenv_values(PROJECT_DIR / '.env.local')}
    for name in ('DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'DEEPSEEK_MODEL'):
        if name in os.environ:
            settings[name] = os.environ[name]
    return settings


class DeepSeekClient:
    def __init__(self, config=None, api_key=None, model=None):
        self.config = Config() if config is None else config
        settings = read_local_settings()
        # 网页可显式传入本次会话的密钥；未传入时读取本地配置。
        self._api_key = (api_key or settings.get('DEEPSEEK_API_KEY') or '').strip()
        self.model = model or settings.get('DEEPSEEK_MODEL') or self.config.model_name
        self.base_url = settings.get('DEEPSEEK_BASE_URL') or self.config.base_url
        self._client = None
        # 保存最近一次成功分类的 SDK 用量对象，不改变 classify 的字典返回结构。
        # API 返回 usage 时，可通过 last_usage.total_tokens 查看本次总 token 数。
        self.last_usage = None

    @property
    def configured(self):
        return bool(self._api_key)

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    # 只对限流和服务器临时错误重试，最多总计 3 次，不重试无效密钥或非法输出。
    # SDK 自带重试在创建时关闭，避免两层重试次数相乘。
    @retry(retry=retry_if_exception_type((RateLimitError, InternalServerError)),
           stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def _request(self, messages):
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.config.max_tokens,
            response_format={'type': 'json_object'},
            # 新闻分类直接输出 JSON 正文，关闭思考以免预算被推理内容占用。
            extra_body={'thinking': {'type': 'disabled'}},
        )

    def classify(self, title):
        """输入一条新闻标题，返回 {'category': 英文类别, 'reason': 分类理由}。"""
        # 重新调用失败时不保留上一次的用量，避免误把旧消耗当成本次结果。
        self.last_usage = None
        if not isinstance(title, str) or not title.strip():
            raise DeepSeekClientError('新闻标题必须是非空字符串。')
        title = title.strip()
        if len(title) > 5000:
            raise DeepSeekClientError('新闻标题不能超过 5000 个字符。')
        if not self.configured:
            raise DeepSeekClientError('未配置 DEEPSEEK_API_KEY，请在 05_LLM/.env 中填写，或使用页面的密钥输入框。')

        if self._client is None:
            # OpenAI SDK 在这里仅作为兼容客户端，实际请求发往 DeepSeek。
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url,
                                  timeout=self.config.timeout, max_retries=0)
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_PROMPT.format(title_json=json.dumps(title, ensure_ascii=False))},
        ]
        try:
            response = self._request(messages)
        except APITimeoutError:
            raise DeepSeekClientError('请求超时，请检查网络或增大 config.py 的 timeout。') from None
        except APIConnectionError:
            raise DeepSeekClientError('无法连接 DeepSeek，请检查网络与 base_url。') from None
        except APIStatusError as error:
            errors = {401: 'API Key 无效，请检查密钥。', 402: '账户余额不足，请检查 DeepSeek 账户额度。',
                      403: '当前密钥没有调用权限。', 429: '请求过于频繁，请稍后重试。'}
            raise DeepSeekClientError(errors.get(error.status_code, f'DeepSeek 请求失败（HTTP {error.status_code}）。')) from None

        if not response.choices:
            raise DeepSeekClientError('模型没有返回分类结果。')
        if response.choices[0].finish_reason == 'length':
            raise DeepSeekClientError('模型输出达到长度上限，请增大 config.py 的 max_tokens 后重试。')
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            raise DeepSeekClientError('模型未返回合法 JSON，请重试。') from None
        # 再做本地校验，不能仅靠提示词保证模型始终返回正确格式。
        if not isinstance(result, dict) or not isinstance(result.get('category'), str) or result['category'] not in CATEGORY_DESCRIPTIONS:
            raise DeepSeekClientError('模型返回了不在十分类列表中的类别，请重试。')
        if not isinstance(result.get('reason'), str) or not result['reason'].strip():
            raise DeepSeekClientError('模型返回的分类理由为空或类型错误，请重试。')
        # usage 位于完整响应 response 上，不在 choices[0].message 上。
        # 某些兼容服务可能不返回 usage，此时保留 None，而不是编造 0。
        self.last_usage = getattr(response, 'usage', None)
        return {'category': result['category'], 'reason': result['reason'].strip()}


def save_result(path, title, result):
    # 仅在 --save 时追加保存新闻与结果，不保存密钥、请求头或完整配置。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'time': datetime.now().isoformat(timespec='seconds'), 'title': title, **result}
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='大肥鱼 · DeepSeek 新闻十分类')
    parser.add_argument('title', nargs='?', help='要分类的新闻标题')
    parser.add_argument('--text', help='也可通过此选项传入标题')
    parser.add_argument('--save', action='store_true', help='追加保存到 outputs/predictions.jsonl')
    args = parser.parse_args(argv)
    if args.title is not None and args.text is not None:
        parser.error('位置参数 title 与 --text 只能使用一个。')
    title = args.text if args.text is not None else args.title
    client = DeepSeekClient()
    if not client.configured:
        print('请在 05_LLM/.env 中设置 DEEPSEEK_API_KEY 后重试，密钥不要发到聊天中。')
        return 1

    def run_once(text):
        result = client.classify(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save:
            output = client.config.output_dir / 'predictions.jsonl'
            save_result(output, text, result)
            print(f'已保存到：{output}')

    try:
        if title is not None:
            run_once(title)
        else:
            print('大肥鱼新闻分类已就绪，输入新闻标题开始；输入 q / quit / 退出 结束。')
            while True:
                try:
                    text = input('\n新闻标题 > ').strip()
                except EOFError:
                    break
                if text.lower() in ('q', 'quit', 'exit', '退出'):
                    break
                if not text:
                    continue
                try:
                    run_once(text)
                except DeepSeekClientError as error:
                    print(f'分类失败：{error}')
    except DeepSeekClientError as error:
        print(f'分类失败：{error}')
        return 1
    except KeyboardInterrupt:
        print('\n已退出。')
    except OSError:
        print('保存结果失败，请检查输出目录权限。')
        return 1
    finally:
        client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
