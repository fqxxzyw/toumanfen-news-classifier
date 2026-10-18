from flask import Flask, request, jsonify
app = Flask(__name__)
from predictor import *
from config import *
conf = Config()

# 创建fasttestPredictor
predictor = FastTextPredictor(str(conf.model_dir/'fasttext_char_autotune.bin'),'char')

# 编写健康接口
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})


@app.route('/predict',methods = ['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({
            'code':400,
            "msg":'No data provided'
        })
    text = data.get('text').strip()

    result = predictor.predict(text)
    return jsonify(result)










if __name__ == '__main__':
    app.run('0.0.0.0',port=8000)
