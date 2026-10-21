import time

from classifer import classify_news
from config import config
import pandas as pd
# news='丈夫1个拥抱 妻子断了5根肋骨'
# result=classify_news(news)
# print(result)

# 10000新闻   已经预测了6000   服务器宕机了    下一次 从头开始，继续上次没跑完的
# 断点续跑

# 先读取一条数据--->查缓存---->调用API---->校验输出---->落盘(逐条保存)
# 定义预测输出的文件
output_dir=config.output_dir / 'dev_predictions.csv'

def load_data(path):
    df=pd.read_csv(path,sep='\t',header=None,names=['news','y_true'],encoding='utf-8')
    # 给每条数据增加rowid
    df.insert(0,'row_id',range(len(df)))
    return df
# 获取已经完成预测的新闻编号
def load_finish_ids():
    # 如果结果文件不存在
    if not output_dir.exists():
        return set()
    # 读取已有的结果
    old_df=pd.read_csv(output_dir)
    # 只把success当做完成
    success_df=old_df[old_df['status']=='success']
    # 返回成功的row_id
    return set(success_df['row_id'].astype(int).tolist())
# 拼接结果
def append_result(result):
    # 将单条的结果转换成DataFrame
    result_df=pd.DataFrame([result])
    # 判断文件是否存在
    file_exists=output_dir.exists() and output_dir.stat().st_size > 0
    # 追加写入
    result_df.to_csv(
        output_dir,
        mode='a',
        header=not file_exists,# 如果已经有表头不需要重新写
        index=False, #不保存dataframe的索引
        encoding='utf-8-sig'
    )
# 批量预测数据
def run_batch():
    # 读取dev数据
    df=load_data(config.dev_path)
    # 测试阶段随机取500条数据进行演示
    # 数据不足 500 条时取全部，避免小样本调试时 sample 报错。
    df=df.sample(n=min(500, len(df)),random_state=42)
    # 获取已经完成数据的id
    finish_ids=load_finish_ids()
    # 逐条处理数据
    for row in df.itertuples(index=False):
        # 判断这条数据是否已经被预测过
        if row.row_id in finish_ids:
            continue
        # 获取当前时间
        start_time=time.perf_counter()
        try:
            # 调用模型预测
            pred=classify_news(row.news)
            # 计算请求耗时
            latency_ms=(time.perf_counter()-start_time)*1000
            # 构造成功的结果
            result={
                'row_id':row.row_id,
                'title':row.news,
                'true_lable':row.y_true,
                # classify_news 把客户端的 category 重命名成了 class_name。
                'pred_label':pred['class_name'],
                'reason':pred['reason'],
                'status':'success',
                'latency_ms':latency_ms,
                # 当前封装只返回 total_usage，没有分别返回输入/输出 token 数。
                # 用 get 兼容可选字段；没有数据时留空，不能伪造为 0。
                'prompt_tokens':pred.get('prompt_tokens'),
                'completion_tokens':pred.get('completion_tokens'),
                'model':config.model_name,
                'error':''
            }
        except Exception as error:
            result = {
                'row_id': row.row_id,
                'title': row.news,
                'true_lable': row.y_true,
                'pred_label': None,
                'reason': None,
                'status': 'fail',
                'latency_ms': None,
                'prompt_tokens': None,
                'completion_tokens': None,
                'model': config.model_name,
                'error':str(error)
            }
        append_result(result)

if __name__ == '__main__':
    #print(load_data(config.dev_path))
    run_batch()
