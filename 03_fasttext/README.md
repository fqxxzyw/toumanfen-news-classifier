# FastText 新闻分类

本阶段实现字符/词语预处理、训练、自动调参、四模型验证、单条预测、API 和网页。

## 运行顺序

从本目录执行，使用 Python 3.10 环境：

```bat
conda activate toumanfen
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

已有 processed 数据可跳过预处理，已有模型可跳过训练。预处理会覆盖六个同名输出；
训练会覆盖四个同名模型，包含两个基础训练与两个 300 秒自动调参搜索。
搜索预算不保证整个训练脚本恰好十分钟完成。

| 路径 | 内容 |
| --- | --- |
| `data/raw/*.txt` | 原始 train/dev/test、类别表 |
| `data/processed/{train,dev,test}_{char,word}.txt` | 六个 FastText 格式文件 |
| `models/fasttext_char.bin` | 字符基础模型 |
| `models/fasttext_word.bin` | 词语基础模型 |
| `models/fasttext_char_autotune.bin` | 字符调参模型 |
| `models/fasttext_word_autotune.bin` | 词语调参模型 |

原始格式：`标题<Tab>编号`；输出格式：`__label__类别 分词文本`。
类别编号对应 class.txt 行号。模型不在 Git 或 Release 附件中，字符调参模型加载需要数 GB 内存。

## 函数与调用关系

| 文件 / 函数 | 输入 | 处理与输出 |
| --- | --- | --- |
| config.py / Config.__init__ | 无 | 定位数据、模型目录并创建输出目录 |
| preprocess.py / load_label | 类别路径 | 编号到英文类别名的字典 |
| preprocess_text | text、mode | char 按字符，word 用 jieba；返回空格分隔文本 |
| build_fasttest_data | 输入/输出路径、标签字典、mode | 拆分标题和标签，预处理后覆盖写入 |
| train.py / train_char、train_word | 配置中的训练数据 | 接收 train_supervised 返回的模型，调用保存 |
| auto_param_tuning | mode、duration，默认 60 秒 | 使用相应 dev 搜索参数，主入口指定 300 秒 |
| save_model | 模型、名字 | 输出 models/名字.bin |
| evaluate.py / evaluate_model | 模型、dev 路径 | 批量 Top-1 预测，输出准确率与分类报告 |
| FastTextPredictor.__init__ | 模型路径、mode | 加载模型，保存预处理模式 |
| FastTextPredictor.predict | 单条 text | 复用预处理，返回 label 与 probability |
| api.py / health | GET /health | 返回 status=OK |
| api.py / predict | POST /predict，JSON text | 调用预测封装，返回类别和概率 |
| app.py 顶层页面 | 文本框和按钮 | 请求 8000 端口，显示预测或请求异常 |

调用顺序：原始数据 → 预处理 → 训练/调参 → 保存 → 评估或预测 → API → 页面。

## API 和页面

两个终端均进入本目录并保持运行：

```bat
python src/api.py
```

```bat
python -m streamlit run src/app.py
```

打开 http://localhost:8501，API 默认加载字符自动调参模型。
请求为 POST http://127.0.0.1:8000/predict，JSON 为：

```json
{"text":"郑州马拉松今日正式开跑"}
```

成功响应包含 label、probability。单条脚本示例为 `python src/predictor.py`。
示例加载字符模型，mode=char；换用词语模型时改为 mode=word。
API 为本地演示，缺失 text、非字符串等异常尚未完整校验，页面请求尚未配置超时。

## 常见问题与指标

- 当前 fastText 不接收 WindowsPath，并且中文绝对路径可能打不开；代码使用 os.path.relpath。
  请从本目录、其 src 或仓库根目录运行，使交给原生库的路径不含中文。
- 当前 fastText 单条 predict(text) 与 NumPy 2 的 copy=False 有兼容问题，改用列表批量接口。
- 自动调参参数是 autotuneValidationFile、autotuneDuration，后者单位为秒。
- Streamlit 需用 `python -m streamlit run`，普通 Python 运行不会启动网页服务。
- RequestsDependencyWarning 属于编码检测依赖问题；requirements 已列出 charset-normalizer，
  已有环境仍警告时应检查该包能否正常导入。

四模型 dev 准确率依次为 87.17%、89.90%、91.43%、91.55%。
每类别 P=TP/(TP+FP)，R=TP/(TP+FN)，F1 为调和平均，support 为真实样本数。
单标签任务总体 P@1、R@1 等于准确率，分类报告中的每类别 P/R 不一定相同。
dev 同时用于调参，这不是最终独立 test 基准；详见 [验证记录](../docs/VALIDATION.md)。
