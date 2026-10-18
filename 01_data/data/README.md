# 中文新闻标题十分类数据

本次下载的是公开的 THUCNews 20 万条新闻标题子集，不是伪造数据。

来源：https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/master/THUCNews/data

下载日期：2026-10-15。保留来源文件内容与原有划分，未分词、未重新清洗或去重。来源数据已经带分类标签，属于标题分类样本，并非新闻全文原始抓取数据。

| 文件 | 条数 | 格式 |
| --- | ---: | --- |
| train.txt | 180000 | 标题 + Tab + 标签 ID |
| dev.txt | 10000 | 标题 + Tab + 标签 ID |
| test.txt | 10000 | 标题 + Tab + 标签 ID |
| class.txt | 10 | 每行一个类别名 |

UTF-8 编码，Tab 是实际制表符。class.txt 的标签从 **0** 开始，即第一行对应标签 0。

| 标签 | 原始类别名 | 中文 |
| --- | --- | --- |
| 0 | finance | 财经 |
| 1 | realty | 房产 |
| 2 | stocks | 股票 |
| 3 | education | 教育 |
| 4 | science | 科技 |
| 5 | society | 社会 |
| 6 | politics | 时政 |
| 7 | sports | 体育 |
| 8 | game | 游戏 |
| 9 | entertainment | 娱乐 |

请使用带 .txt 后缀的四个原始数据文件。运行 `src/02_data_process.py` 会生成三个 CSV；这些生成文件不提交到 Git，可以重新处理获得。

`stopwords.txt` 是项目已有的停用词表，按空白字符拆分。其具体上游来源尚未记录，不能据此宣称某一套词表的授权许可。
