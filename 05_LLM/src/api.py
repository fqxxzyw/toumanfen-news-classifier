"""DeepSeek 新闻分类 HTTP 接口；本地配置密钥，导入时不调用 API。"""

from flask import Flask, jsonify, request

from classifer import classify_news
from deepseek_client import DeepSeekClientError

app = Flask(__name__)


@app.get('/health')
def health():
    return jsonify({'status': 'OK'})


@app.post('/predict')
def predict():
    # silent=True 让缺失或损坏的 JSON 也使用相同的 JSON 错误格式返回。
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': '请求体必须是 JSON 对象。'}), 400
    title = data.get('text')
    if not isinstance(title, str) or not title.strip():
        return jsonify({'error': 'text 必须是非空字符串。'}), 400
    title = title.strip()
    if len(title) > 5000:
        return jsonify({'error': 'text 不能超过 5000 个字符。'}), 400
    try:
        result = classify_news(title)
        return jsonify({'code': 0, 'data': {
            'title': title,
            # classify_news 使用 class_name，HTTP 响应对外仍使用 category。
            'category': result['class_name'],
            'reason': result['reason'],
        }})
    except DeepSeekClientError as error:
        return jsonify({'error': str(error)}), 502
    except Exception:
        # 不向 HTTP 调用方回传内部异常、请求头或本地配置。
        return jsonify({'error': '分类服务暂时不可用。'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
