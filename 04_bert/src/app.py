"""Streamlit 新闻分类页面：只调用 HTTP API，不在页面进程中加载 BERT。"""

import os
from time import perf_counter

import requests
import streamlit as st


# 可通过环境变量切换接口地址，默认使用同一台电脑上的 Flask 服务。
API_URL = os.environ.get('NEWS_API_URL', 'http://127.0.0.1:8000').rstrip('/')

# 模型保留训练时的英文标签，页面提供对应的中文名称用于展示。
CLASS_NAMES = {
    'finance': '财经', 'realty': '房产', 'stocks': '股票',
    'education': '教育', 'science': '科技', 'society': '社会',
    'politics': '时政', 'sports': '体育', 'game': '游戏',
    'entertainment': '娱乐',
}
EXAMPLES = {
    '体育新闻': '中国队在总决赛中战胜对手，成功夺得冠军。',
    '科技新闻': '新一代人工智能芯片发布，计算性能和能效进一步提升。',
    '财经新闻': '央行公布最新货币政策，支持实体经济稳步发展。',
}


@st.cache_data(ttl=5, show_spinner=False)
def get_service_status(api_url):
    # 短暂缓存健康检查，减少页面重绘时的重复请求；不缓存用户的预测文本。
    try:
        response = requests.get(f'{api_url}/health', timeout=(2, 3))
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) and data.get('status') == 'ok' else None
    except (requests.RequestException, ValueError):
        return None


def request_prediction(text):
    # requests 在 Streamlit 的 Python 进程中运行，浏览器不直接调用 Flask。
    # timeout 分别限制连接时间和等待响应的时间，避免服务异常时一直转圈。
    response = requests.post(
        f'{API_URL}/predict', json={'text': text}, timeout=(3, 60)
    )
    try:
        data = response.json()
    except ValueError as error:
        raise ValueError('预测服务没有返回有效的 JSON，请检查接口地址。') from error
    if not response.ok:
        message = data.get('error') if isinstance(data, dict) else None
        raise ValueError(message or f'预测服务返回错误：HTTP {response.status_code}')
    if not isinstance(data, dict) or not {'pred', 'class', 'score'} <= data.keys():
        raise ValueError('预测服务返回的数据缺少类别或分数字段。')
    return data


def fill_example(text):
    # 按钮回调会在页面重新执行前更新输入框，并清除上一次的结果。
    st.session_state['news_text'] = text
    st.session_state.pop('prediction', None)


def main():
    st.set_page_config(page_title='BERT 新闻分类', page_icon='📰', layout='wide')
    st.title('📰 新闻分类助手')
    st.caption('输入新闻标题或短文本，让 BERT 判断它所属的新闻类别。')

    status = get_service_status(API_URL)
    with st.sidebar:
        st.header('模型与服务')
        if status:
            st.success('预测服务已连接')
            st.caption(f"运行设备：{status['device']}")
            st.caption(f"最大输入长度：{status['max_length']} 个词元")
            st.caption('超长文本会按照训练时的长度设置截断。')
            classes = status['class_list']
        else:
            st.warning('预测服务未连接')
            st.caption('请先启动 API 服务，再刷新连接状态。')
            classes = list(CLASS_NAMES)
        if st.button('刷新连接状态', use_container_width=True):
            get_service_status.clear()
            st.rerun()
        st.divider()
        st.subheader('支持的类别')
        st.write(' · '.join(CLASS_NAMES.get(label, label) for label in classes))

    editor, output = st.columns([1.3, 1], gap='large')
    with editor:
        st.subheader('输入新闻')
        st.caption('可先选择一条示例，再替换成自己的文本。')
        for column, (label, example) in zip(st.columns(3), EXAMPLES.items()):
            column.button(label, on_click=fill_example, args=(example,), use_container_width=True)

        # 表单让输入和提交成为一次操作，打字时不会不断触发预测。
        with st.form('prediction_form'):
            text = st.text_area(
                '新闻内容', key='news_text', height=230, max_chars=5000,
                placeholder='例如：球队在总决赛中获得冠军，球迷现场欢呼。',
            )
            submitted = st.form_submit_button(
                '开始预测', type='primary', use_container_width=True
            )

        if submitted:
            # 每次提交先清除旧结果，避免请求失败时仍展示上一次的预测。
            st.session_state.pop('prediction', None)
            if not text.strip():
                st.warning('请先输入新闻内容。')
            else:
                try:
                    started = perf_counter()
                    with st.spinner('正在分析新闻…'):
                        result = request_prediction(text.strip())
                    st.session_state['prediction'] = {
                        'result': result, 'text': text.strip(),
                        'elapsed': perf_counter() - started,
                    }
                except requests.Timeout:
                    st.error('预测请求超时，请稍后重试。')
                except requests.ConnectionError:
                    st.error('无法连接预测服务，请确认 API 已启动。')
                except (requests.RequestException, ValueError) as error:
                    st.error(str(error))

    with output:
        st.subheader('最近一次预测')
        saved = st.session_state.get('prediction')
        if saved:
            result = saved['result']
            label = CLASS_NAMES.get(result['class'], result['class'])
            with st.container(border=True):
                st.metric('预测类别', label)
                st.caption(f"标签：{result['class']}  ·  类别 ID：{result['pred']}")
                st.metric('模型分数', f"{result['score']:.2%}")
                st.progress(float(result['score']))
                st.caption('分数为模型对该类别的 softmax 概率。')
                st.caption(f"本次请求耗时：{saved['elapsed']:.2f} 秒")
            with st.expander('查看本次输入与接口结果'):
                st.write(saved['text'])
                st.json(result)
        else:
            st.info('输入新闻并点击“开始预测”，这里会显示类别和模型分数。')


if __name__ == '__main__':
    main()
