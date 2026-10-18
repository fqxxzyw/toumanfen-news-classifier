"""步骤 05：启动新闻分类 HTTP API，端口 8001。"""
from flask import Flask, request, jsonify
from prediction import predict_fun
from config import Config
app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict_api():
    # 接受用户的数据
    data = request.get_json(silent=True)
    '''
    常见状态码：
    200 OK
    4XX 路径或者提交方式
    5XX 服务器内部问题
    3XX 重定向问题
    '''
    if not isinstance(data, dict) or not isinstance(data.get('text'), str) or not data['text'].strip():
        return jsonify({'error':'Invalid data'}),400

    # 调用预测函数
    data['text'] = data['text'].strip()
    result = predict_fun(data)
    return jsonify({'result':result}),200
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8001)
