# 文件迁移对照

整理日期：2026-10-17。保留现有实现和学习注释，调整入口名称、相互导入及路径配置。

| 原文件 | 新位置 | 用途 |
| --- | --- | --- |
| `src/eda.py` | `src/01_eda.py` | 原始数据分析 |
| `src/data_process.py` | `src/02_data_process.py` | 清洗与分词 |
| `src/train.py` | `src/03_train.py` | 分批训练与内部评估 |
| `src/predict.py` | `src/04_evaluate.py` | 独立验证集评估，补充指标和结果导出 |
| `src/predict_api.py` | `src/05_predict_api.py` | 分类 API |
| `src/app.py` | `src/06_app.py` | Streamlit 页面 |
| `src/predict_fun.py` | `src/prediction.py` | 共用预测模块 |
| `src/config.py` | 原位置 | 共用路径配置 |
| `src/training_utils.py` | 原位置 | 随机森林和预测进度工具 |
| 根目录 `01_load_data.py` | 本地 `预习/01_load_data.py` | 保留原练习，修改为脚本相对路径 |
| 根目录 `02_data_cleaning.py` | 本地 `预习/02_data_cleaning.py` | 保留原练习，输出到预习区 |
| `src/predict_test.py` | 本地 `预习/07_predict_test.py` | 请求演示，指向预习 API 8002 |
| `src/flask_api.py` | 本地 `预习/08_http_demo.py` | GET / POST / 加法演示 |
| `src/index.html` | 本地 `预习/index.html` | 配合 HTTP 演示使用 |

本地预习区还从正式代码复制了步骤 03～06 及配套模块。正式代码与练习代码分别演进，不通过自动同步覆盖用户的后续练习。预习完整操作说明只保存在本地 `预习/README.md` 中。

正式 API 的导入已更新为 `from prediction import predict_fun`，页面启动提示已更新为编号后的入口。旧的 PyCharm 运行配置可能仍指向原文件，请重新选择对应的新入口。
