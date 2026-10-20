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
- `src/config.py`：继续使用原有模型名称、地址、128 tokens 输出上限和 30 秒超时；页面可单独调整输出上限。
- `assets/dafeiyu.png`：通过内置 ImageGen 生成的主题插画，生成提示词见 `assets/README.md`。

SDK 只用于兼容 DeepSeek 接口，请求发送至配置的 DeepSeek 地址，不使用 OpenAI 服务的密钥。[接口格式参考 DeepSeek 官方文档](https://api-docs.deepseek.com/api/create-chat-completion/)。主题是社区二创，不是官方客户端；[形象方向参考](https://github.com/ERUIHNIYHBKBNF/dafeiyu-persona)。

## 离线检查

```powershell
python -m unittest discover -s tests -p test_deepseek.py -v
```

检查单条与交互命令行、请求格式、非法 JSON/类别/空输入、输出截断、缺少密钥及页面提交。不使用真实密钥，不发起外部请求。真实分类还需要有效的 DeepSeek API Key 和账户额度。
