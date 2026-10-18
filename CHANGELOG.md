# 版本记录

## v0.2.0 · 2026-10-18

完成 FastText 本地分类流程，新增 BERT 数据加载阶段。

### FastText

- char/word 两模式生成六份数据，覆盖写入，标签位于行首，避免重复追加。
- 字符/词语基础训练及各 300 秒自动调参，保存四个模型。
- 修复 WindowsPath 类型、中文绝对路径、未接收模型返回值和 autotuneDuration 拼写问题。
- 四模型 dev 评估、准确率和分类报告；dev 准确率为 87.17%、89.90%、91.43%、91.55%。
- 批量预测避开 NumPy 2 兼容问题，修复 Config 实例化、预测标签格式及预处理参数名。
- 单条预测、Flask /health 和 /predict、Streamlit 页面与正确启动方式。

### BERT

- 新增配置、数据、类别表、分词器配置及词表。
- 实现 load_news_data(file_path)、Dataset、批次 padding/truncation、labels 张量与 DataLoader。
- 修复函数参数不一致、news_path/max_len 配置名称错误及分词器相对路径。
- 读取 180,000 条训练数据并验证首个 batch，尚未实现微调训练与评估。

### 文档与发布

- 根目录新增 VERSION、CHANGELOG、发布与验证记录，补充阶段教程、函数说明和依赖文件。
- dev 用于调参，成绩不代表独立 test 指标。
- 源码发布不含本地模型和大型 BERT 权重，本地文件保留。
- 详细步骤见 [v0.2.0 发布说明](docs/releases/v0.2.0.md)。

## v0.1.0 · 2026-10-17

首次发布随机森林分类基线、六步流程、API、页面和完整开发教程。
原始记录见 [随机森林 CHANGELOG](02_rf/CHANGELOG.md) 和
[v0.1.0 发布说明](02_rf/docs/releases/v0.1.0.md)。
