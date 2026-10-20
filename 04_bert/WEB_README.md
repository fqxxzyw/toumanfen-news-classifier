# 新闻分类页面与 API

数据流：Streamlit 页面（`src/app.py`）→ HTTP 请求 → Flask API（`src/api.py`）→ `predictor.py` → 本地 BERT 模型。

模型仅在 API 启动时加载一次，页面不重复加载模型。使用现有 `models/best_model`，无需重新训练。

## 启动

在 PyCharm 终端或 Anaconda Prompt 中进入 `04_bert` 目录，激活环境并安装依赖：

```powershell
conda activate toumanfen
cd D:\soft\python\NLP\投满分项目\04_bert
python -m pip install -r requirements.txt
```

第一个终端启动 API，并保持运行：

```powershell
python src/api.py
```

等待模型加载完成并显示 `Running on http://127.0.0.1:8000`。也可以在 PyCharm 中直接运行 `src/api.py`。

第二个终端进入同一目录、激活同一个环境，启动页面：

```powershell
python -m streamlit run src/app.py --server.address 127.0.0.1
```

浏览器打开 <http://127.0.0.1:8501>。选择示例或输入新闻，点击“开始预测”。页面展示中文类别、原始英文标签、类别 ID、模型分数和请求耗时。

`app.py` 要用 Streamlit 命令启动，不能用普通的 `python src/app.py` 启动网页。
关闭时分别在两个终端按 `Ctrl+C`。若端口已被占用，先停止之前启动的对应服务。

## 接口

| 方法 | 地址 | 用途 |
| --- | --- | --- |
| GET | `http://127.0.0.1:8000/` | 接口简介 |
| GET | `http://127.0.0.1:8000/health` | 查询模型状态、设备、最大词元长度和类别列表 |
| POST | `http://127.0.0.1:8000/predict` | 预测一条新闻 |

请求头使用 `Content-Type: application/json`，请求体：

```json
{"text": "球队在总决赛中获得冠军"}
```

成功返回示例（具体类别与分数以模型实际输出为准）：

```json
{"pred": 7, "class": "sports", "score": 0.98}
```

`pred` 是类别 ID，`class` 是模型的原始类别名称，`score` 是该类别的 softmax 概率。页面将英文标签转换为中文展示。它不等于模型在测试集上的准确率。

空文本、非字符串、缺少字段、格式错误或超过 5000 字符的输入会返回 HTTP 400 和 `{"error": "错误说明"}`；请求体超过 64 KiB 返回 HTTP 413。文本仍会按训练时保存的 `max_length` 截断，适合输入新闻标题或短文本。

Python 调用示例：

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "球队在总决赛中获得冠军"},
    timeout=60,
)
response.raise_for_status()
print(response.json())
```

页面默认连接本机的 8000 端口。若需要连接其他 API 地址，在启动页面前设置环境变量（PowerShell）：

```powershell
$env:NEWS_API_URL = "http://127.0.0.1:8000"
python -m streamlit run src/app.py --server.address 127.0.0.1
```

## 分享给同学

保留项目中的 `src/config.py`、`src/predictor.py`、`src/api.py`、`src/app.py`、`requirements.txt` 和整个 `models/best_model` 目录，保持相对目录结构即可。模型目录需包含权重、`config.json`、分词器文件和 `metadata.json`。仅做预测时无需携带训练数据和原始预训练模型。

## 检查代码

从 `04_bert` 目录执行以下命令，运行接口与页面的六项自动检查；检查使用轻量预测器，不加载真实权重：

```powershell
python -m unittest discover -s tests -p test_web.py -v
```

覆盖正常预测、非法输入、错误请求、示例填充、空文本、服务断连与超时提示。

实现参考：[Flask 应用工厂](https://flask.palletsprojects.com/en/stable/patterns/appfactories/)、[Streamlit 表单](https://docs.streamlit.io/develop/concepts/architecture/forms)。
