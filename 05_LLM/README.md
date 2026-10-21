# 大肥鱼的新闻小站 🐋

使用 DeepSeek 进行新闻十分类，复用同一个客户端提供命令行与大肥鱼主题 Streamlit 页面。蓝白海洋配色、鲸鱼娘插画、中文类别和分类理由；保留现有 `prompts.py` 的新闻分类任务。

## 先配置密钥

仓库只提供空值样例 `.env.example`。首次使用时在 `05_LLM` 目录复制为 `.env`；如果已有本地 `.env`，直接编辑原文件：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在本机填写自己的 DeepSeek API Key：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

真实密钥不要发到聊天或提交 Git。也可通过系统环境变量配置，或仅在页面侧栏输入。页面输入不会自动写入磁盘。系统环境变量优先于 `.env.local`，后者优先于 `.env`。

## 命令行运行

激活环境，进入 `05_LLM`：

```powershell
conda activate toumanfen
cd 05_LLM
```

依赖已在当前 `toumanfen` 环境安装；新电脑可执行 `python -m pip install -r requirements.txt`。

交互模式，可以连续输入新闻标题，输入 `q`、`quit`、`exit` 或 `退出` 结束：

```powershell
python src/deepseek_client.py
```

单条模式（两种写法任选其一）：

```powershell
python src/deepseek_client.py "球队在总决赛中获得冠军"
python src/deepseek_client.py --text "股票指数上涨，多只个股涨停"
```

输出为分类结果 JSON，例如 `{"category": "sports", "reason": "新闻的核心是体育比赛结果"}`，实际以模型返回为准。

加 `--save` 可追加保存到 `outputs/predictions.jsonl`，每行一条记录，包含时间、标题、类别与理由：

```powershell
python src/deepseek_client.py --text "球队夺冠" --save
```

在 PyCharm 中直接运行 `src/deepseek_client.py` 会进入交互模式；通过运行参数也可传入标题。

## 开发集抽样预测与断点续跑

`src/prodict-one.py` 保留当前文件名，实际用途是批量预测。当前教学版固定用随机种子 42 从 dev.txt 抽样最多 500 条，结果逐条追加到 `outputs/dev_predictions.csv`。运行会发起真实 API 请求，先确认本地密钥与账户额度：

```powershell
python src/prodict-one.py
```

当前入口不接收 `--limit`、`--input` 等参数。要用少量记录调试，可先准备小数据文件并调整 `config.py` 的 `dev_path`；不足 500 条时会取全部。

重复运行只跳过已有 `success` 记录；`fail` 记录会重试。row_id 在抽样前分配，运行期间不要重排或修改原始数据；更换数据源或抽样参数时，先备份旧 CSV，再使用新的结果文件。当前教学版不会自动验证历史文件与数据来源是否匹配，也没有连续失败自动停止功能。

结果字段保留当前 CSV 格式：row_id、title、true_lable、pred_label、reason、status、latency_ms、prompt_tokens、completion_tokens、model、error。其中 `true_lable` 是历史拼写，存储数字标签；`pred_label` 是英文类别。客户端封装未返回分项 token 用量时，对应两列留空，不伪造为 0。

逐条落盘使已完成结果可以保留；同一结果文件不要同时开多个进程写入。CSV 中可包含同一新闻的多次尝试，评估应先筛选成功，再按 row_id 取最后一次成功。仅处理抽样数据不代表完成全部开发集评估。

## 离线评估已有结果

```powershell
python src/evalute.py
python src/evalute.py --input outputs/dev_predictions.csv
```

评估兼容 `true_lable`、`true_id`、`true_label`，按 class.txt 的行号统一数字标签和英文类别。输出成功样本数、成功率、Accuracy、宏平均 Precision/Recall/F1 和各类报告；没有成功记录时会给出明确提示，不会伪造分数。宏平均覆盖全部类别，缺失类别对应的零分母按 0 处理。

## HTTP API

```powershell
python src/api.py
```

服务监听 `127.0.0.1:5001`，提供 `GET /health` 与 `POST /predict`。预测请求体为 `{"text":"球队夺冠"}`，成功响应为 `{"code":0,"data":{"title":"球队夺冠","category":"sports","reason":"..."}}`。输入错误返回 400，外部分类服务异常返回 502，内部错误返回 500。健康检查不调用外部 API，预测需要有效的本地 DeepSeek 配置。

## 主题页面

```powershell
python -m streamlit run src/app.py --server.address 127.0.0.1 --server.port 8502
```

浏览器打开 <http://127.0.0.1:8502>。填写侧栏密钥或使用 `.env`，输入标题并点击“开始分类”。支持示例填充、分类理由、原始 JSON、当前会话记录导出。仅点击分类时请求模型。页面不展示未经校准的“置信度”。

页面会话记录不会自动落盘，刷新后可能丢失，需要保留时点击导出。客户端仅重试限流和临时服务错误，最多总计三次；无效密钥、余额不足或非法 JSON 不会被反复重试。

## 代码说明

- `src/deepseek_client.py`：`DeepSeekClient.classify(title)`、配置加载、JSON 与类别校验、交互/单条命令行入口。
- `src/prompts.py`：十类定义、系统提示词及用户提示词模板。
- `src/app.py`：主题页面，复用客户端，不另写一套分类逻辑。
- `src/classifer.py`：单条分类封装，将 category 转为 class_name，并读取总 token 用量。
- `src/prodict-one.py`：开发集批量预测与成功记录跳过。
- `src/evalute.py`：已有预测 CSV 的离线分类评估。
- `src/api.py`：本地 Flask 健康检查与预测接口。
- `src/config.py`：继续使用原有模型名称、地址、128 tokens 输出上限和 30 秒超时；页面可单独调整输出上限。
- `assets/dafeiyu.png`：通过内置 ImageGen 生成的主题插画，生成提示词见 `assets/README.md`。

SDK 只用于兼容 DeepSeek 接口，请求发送至配置的 DeepSeek 地址，不使用 OpenAI 服务的密钥。[接口格式参考 DeepSeek 官方文档](https://api-docs.deepseek.com/api/create-chat-completion/)。主题是社区二创，不是官方客户端；[形象方向参考](https://github.com/ERUIHNIYHBKBNF/dafeiyu-persona)。

## 离线检查

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

检查单条与交互命令行、请求格式、非法 JSON/类别/空输入、输出截断、缺少密钥及页面提交。不使用真实密钥，不发起外部请求。真实分类还需要有效的 DeepSeek API Key 和账户额度。
