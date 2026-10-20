"""大肥鱼新闻分类页面，与命令行复用 DeepSeekClient。"""

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import streamlit as st

from config import Config
from deepseek_client import DeepSeekClient, DeepSeekClientError, read_local_settings
from prompts import CATEGORY_DESCRIPTIONS

PROJECT_DIR = Path(__file__).resolve().parents[1]
MASCOT = PROJECT_DIR / 'assets/dafeiyu.png'
CATEGORY_NAMES = {
    'finance': '财经', 'realty': '房产', 'stocks': '股票', 'education': '教育',
    'science': '科技', 'society': '社会', 'politics': '时政', 'sports': '体育',
    'game': '游戏', 'entertainment': '娱乐',
}
EXAMPLES = {
    '🏀 体育赛事': '中国队在总决赛中战胜对手，夺得冠军。',
    '💻 科技动态': '新一代人工智能芯片发布，计算性能和能效进一步提升。',
    '📈 股市行情': '上证指数收涨，半导体板块多只个股涨停。',
}


def fill_example(text):
    st.session_state['news_title'] = text
    st.session_state.pop('prediction', None)


def main():
    st.set_page_config(page_title='大肥鱼的新闻小站', page_icon='🐋', layout='wide')
    st.markdown(f'<style>{(PROJECT_DIR / "assets/theme.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
    config = Config()
    settings = read_local_settings()
    st.session_state.setdefault('records', [])

    with st.sidebar:
        st.markdown('### 🐋 大肥鱼的新闻小站')
        st.caption('A LITTLE WHALE · A LITTLE WISDOM')
        st.image(str(MASCOT), width=180)
        st.markdown('**饭可以晚点吃，新闻先帮你分。**')
        st.caption('DeepSeek 社区二创主题 · 非官方客户端')
        st.divider()
        with st.expander('连接设置', expanded=not bool(settings.get('DEEPSEEK_API_KEY'))):
            # 本地已有密钥时不把它作为输入框默认值，避免发送到浏览器。
            api_key = st.text_input('DeepSeek API Key', type='password',
                                    help='可留空使用 .env 或系统环境变量中的密钥。')
            model = st.text_input('模型名称', value=settings.get('DEEPSEEK_MODEL') or config.model_name)
            max_tokens = st.select_slider('输出长度上限', options=[128, 256, 512, 1024], value=256)
        configured = bool(api_key.strip() or settings.get('DEEPSEEK_API_KEY'))
        if configured:
            st.success('密钥已就绪')
            st.caption('点击“开始分类”后才调用模型。')
        else:
            st.info('填好密钥，就可以叫大肥鱼开工啦。')
        st.divider()
        st.markdown('**十种新闻，各就各位**')
        st.write(' · '.join(CATEGORY_NAMES.values()))

    st.markdown('<div class="sea-eyebrow">DEEP BLUE / NEWS LAB</div>', unsafe_allow_html=True)
    intro, illustration = st.columns([1.8, 1], vertical_alignment='center')
    with intro:
        st.markdown('<h1 class="sea-title">新闻交给我，<br>你负责投喂就好。</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sea-subtitle">把一条新闻放进小海湾，大肥鱼帮你找到它的类别，<br>再认真说说为什么。咕噜——现在开始？</p>', unsafe_allow_html=True)
        st.markdown('<span class="sea-tag">10 种新闻类别</span> <span class="sea-tag">类别 + 分类理由</span>', unsafe_allow_html=True)
    with illustration:
        st.image(str(MASCOT), width=260)

    editor, result_panel = st.columns([1.25, 1], gap='large')
    with editor:
        st.subheader('给大肥鱼一条新闻')
        for column, (label, example) in zip(st.columns(3), EXAMPLES.items()):
            column.button(label, on_click=fill_example, args=(example,), use_container_width=True)
        with st.form('classify_form'):
            title = st.text_area('新闻标题或短文本', key='news_title', height=160, max_chars=5000,
                                 placeholder='例如：球队在总决赛中获得冠军，球迷现场欢呼。')
            submitted = st.form_submit_button('🐋 开始分类', type='primary', use_container_width=True)
        if submitted:
            st.session_state.pop('prediction', None)
            if not title.strip():
                st.warning('先给我一条新闻嘛，空碗可分不出类别。')
            elif not configured:
                st.error('请先在左侧填写 DeepSeek API Key，或配置本地 .env。')
            else:
                # 每次创建轻量客户端；首次分类时才创建连接，结束后及时关闭。
                client = DeepSeekClient(config=replace(config, max_tokens=max_tokens),
                                        api_key=api_key, model=model.strip())
                try:
                    started = perf_counter()
                    with st.spinner('大肥鱼正在认真读新闻…'):
                        result = client.classify(title)
                    record = {'title': title.strip(), **result,
                              'elapsed': round(perf_counter() - started, 2)}
                    st.session_state['prediction'] = record
                    st.session_state['records'].append(record)
                except DeepSeekClientError as error:
                    st.error(str(error))
                finally:
                    client.close()

    with result_panel:
        st.subheader('大肥鱼的判断')
        record = st.session_state.get('prediction')
        if record:
            with st.container(border=True):
                st.metric('新闻类别', CATEGORY_NAMES[record['category']])
                st.caption(f"原始标签：{record['category']} · 耗时 {record['elapsed']:.2f} 秒")
                st.markdown('**为什么这么分？**')
                st.write(record['reason'])
                st.caption('咕噜，这次也有好好干活！')
            with st.expander('查看本次输入与 JSON 结果'):
                st.write(record['title'])
                st.json({'category': record['category'], 'reason': record['reason']})
        else:
            st.info('输入新闻，点击“开始分类”，这里就会浮出结果。')

    with st.expander('查看十分类的范围'):
        st.table([{'类别': CATEGORY_NAMES[label], '标签': label, '范围': description}
                  for label, description in CATEGORY_DESCRIPTIONS.items()])
    if st.session_state['records']:
        with st.sidebar:
            # 只导出分类记录，不包含密钥和连接设置。
            st.download_button('↓ 带走本次分类记录',
                               data=json.dumps(st.session_state['records'], ensure_ascii=False, indent=2),
                               file_name=f'dafeiyu-{datetime.now():%Y%m%d-%H%M%S}.json',
                               mime='application/json', use_container_width=True)
            if st.button('清空分类记录', use_container_width=True):
                st.session_state['records'] = []
                st.session_state.pop('prediction', None)
                st.rerun()


if __name__ == '__main__':
    main()
