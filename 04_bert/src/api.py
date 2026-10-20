"""Flask 预测接口：运行 python src/api.py 后，监听本机 8000 端口。"""

from threading import Lock

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from predictor import BertNewsClassifierPredictor


def create_app(predictor=None):
    # 应用工厂：模型只在创建应用时加载一次，每次请求复用同一个预测器。
    # predictor 参数允许测试时传入轻量替身，正常运行时不需要传。
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
    if predictor is None:
        predictor = BertNewsClassifierPredictor()

    # 多个页面同时请求时，按顺序使用模型，避免同时推理造成内存占用陡增。
    inference_lock = Lock()

    @app.get('/')
    def index():
        return jsonify({
            'name': 'BERT 新闻分类 API',
            'health': 'GET /health',
            'predict': 'POST /predict',
            'example': {'text': '球队在总决赛中获得冠军'},
        })

    @app.get('/health')
    def health():
        # 应用创建成功才会开始提供服务，所以这里返回 ok 表示模型已经加载。
        return jsonify({
            'status': 'ok',
            'device': str(predictor.device),
            'max_length': predictor.max_len,
            'class_list': predictor.class_list,
        })

    @app.post('/predict')
    def predict():
        # 请求体应为 JSON 对象，例如 {"text": "球队在总决赛中获得冠军"}。
        # silent=True 让格式错误的 JSON 返回 None，由下面统一给出中文提示。
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': '请求体必须是 JSON 对象，并包含 text 字段。'}), 400

        text = payload.get('text')
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'text 必须是非空字符串。'}), 400
        text = text.strip()
        if len(text) > 5000:
            return jsonify({'error': '文本不能超过 5000 个字符。'}), 400

        try:
            with inference_lock:
                result = predictor.predict_one(text)
        except Exception:
            # 详细异常写入后端日志，前端接收可读的错误信息。
            app.logger.exception('新闻分类预测失败')
            return jsonify({'error': '预测失败，请查看 API 终端中的错误信息。'}), 500

        # 沿用 predictor.py 的返回结构：pred 是类别 ID，class 是类别名称，
        # score 是预测类别的 softmax 概率，可直接被页面或其他程序使用。
        return jsonify(result)

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        # 404、405、413 等 HTTP 错误也使用 JSON，避免调用方收到 HTML 错误页。
        message = '请求内容过大。' if error.code == 413 else error.description
        return jsonify({'error': message}), error.code

    return app


if __name__ == '__main__':
    # 关闭调试与自动重载，避免重复加载约 390 MB 的模型。
    # 仅监听本机，Streamlit 后端通过 HTTP 调用这里的接口。
    app = create_app()
    app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)
