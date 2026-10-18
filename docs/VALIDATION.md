# v0.2.0 验证记录

日期：2026-10-18。环境：Windows、Anaconda toumanfen、Python 3.10.22。

## 环境版本

当前安装元数据：fasttext-wheel 0.9.2、NumPy 2.2.5、scikit-learn 1.7.2、
jieba 0.42.1、Flask 3.1.3、Streamlit 1.64.0、requests 2.34.2、
torch 2.14.1、transformers 5.19.0。
这记录已有环境，不代表已经在干净环境重建全部依赖；GPU 版本应按设备选择。

## 实际执行的检查

| 范围 | 内容 | 结果 |
| --- | --- | --- |
| 语法 | 03_fasttext、04_bert 的 9 个 Python 文件 AST 解析 | 通过 |
| 预处理 | 临时样本的 char/word、标签与重复运行覆盖 | 通过 |
| 基础训练/保存 | 临时小样本、低维参数，调用两个训练函数并保存真实模型 | 通过，不覆盖已有模型 |
| 中文路径 | 相同样本的中文绝对、英文相对和英文绝对路径训练/保存 | 当前 fastText 中文绝对路径失败，后两种通过 |
| 调参传参 | 两模式经过实际 Python 参数解析，在原生训练入口前截停 | 参数名、300 秒预算、训练/验证路径检查通过；未重新执行搜索 |
| 四模型评估 | 本地四个已有模型各评估对应 dev 10,000 条 | 均成功，见下表 |
| 单条预测 | 字符调参模型、char 预处理、批量预测接口 | 返回类别和概率 |
| API | GET http://127.0.0.1:8000/health | status=OK |
| Streamlit | 模块启动，GET http://127.0.0.1:8501/_stcore/health | ok；不是完整浏览器交互测试 |
| 发布复查 | AppTest 点击预测，requests.post 注入真实 Flask test_client，加载已有字符调参模型 | /health、/predict、结果中的 label/probability 和页面 JSON 显示通过；不依赖存活的后台服务 |
| BERT | 本地 tokenizer、读取 train、取首个非随机 batch | 180,000 条，input_ids=(16,26)，labels=(16,) |

## FastText dev 指标

| 模型 | 准确率 | 样本数 |
| --- | ---: | ---: |
| fasttext_char | 0.8717 | 10,000 |
| fasttext_word | 0.8990 | 10,000 |
| fasttext_char_autotune | 0.9143 | 10,000 |
| fasttext_word_autotune | 0.9155 | 10,000 |

dev 也用于自动调参，因此不属于最终独立 test 成绩。
发布整理使用已有模型，没有重新全量训练或覆盖模型；其他环境重训结果可能不同。

## 当前限制

- 相对路径方案要求工作目录位于项目内，使交给 fastText 的路径不含中文。
- 当前 requests 报过编码检测依赖警告，这与 Streamlit 启动方式不同。
- BERT 只验证数据输入，未训练，不报告分类准确率。
- API 异常输入校验、网页请求超时、完整浏览器交互、负载和生产部署未完善。
- 未执行最终 test 基准、量化、蒸馏或资源优化。
