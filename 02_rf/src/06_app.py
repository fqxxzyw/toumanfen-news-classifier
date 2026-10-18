"""步骤 06：python -m streamlit run src/06_app.py。"""

import requests
import streamlit as st


API_URL = 'http://127.0.0.1:8001/predict'
CLASS_NAMES = {
    'finance': '财经', 'realty': '房产', 'stocks': '股票',
    'education': '教育', 'science': '科技', 'society': '社会',
    'politics': '时政', 'sports': '体育', 'game': '游戏',
    'entertainment': '娱乐',
}


def request_prediction(text):
    """调用现有 Flask 接口，返回结果；异常交给页面提示。"""
    response = requests.post(
        API_URL, json={'text': text}, timeout=(5, 60),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError('接口返回的不是 JSON 对象')
    result = payload.get('result')
    if not isinstance(result, dict):
        raise ValueError('接口响应缺少 result 对象')
    label = result.get('label')
    if not isinstance(label, str) or not label.strip():
        raise ValueError('接口响应缺少有效的分类标签 label')
    return {'text': text, 'label': label.strip()}


def main():
    st.set_page_config(page_title='新闻分类器', page_icon='📰')
    st.title('📰 新闻分类器')
    st.caption('输入一条中文新闻标题，识别它所属的新闻类别。')

    with st.sidebar:
        st.subheader('支持的类别')
        st.write('、'.join(CLASS_NAMES.values()))
        st.info('模型根据新闻标题训练，建议输入简短、完整的新闻标题。')
        with st.expander('启动说明'):
            st.write('先启动分类接口，再启动此页面。在项目根目录执行：')
            st.code('python src/05_predict_api.py\npython -m streamlit run src/06_app.py', language='bash')

    with st.form('news_form'):
        text = st.text_area(
            '请输入要分类的新闻内容',
            placeholder='例如：中国队在世界锦标赛中夺得冠军',
            max_chars=2000,
            height=150,
        )
        submitted = st.form_submit_button('开始分类', type='primary')

    if submitted:
        # 每次提交先清除上次结果，避免失败时显示旧分类。
        st.session_state.pop('prediction', None)
        clean_text = text.strip()
        if not clean_text:
            st.warning('请先输入新闻内容，不能只输入空格。')
        else:
            try:
                with st.spinner('正在识别新闻类别…'):
                    st.session_state['prediction'] = request_prediction(clean_text)
            except requests.exceptions.Timeout:
                st.error('分类请求超时，请稍后重试。首次加载模型可能需要一些时间。')
            except requests.exceptions.ConnectionError:
                st.error('无法连接分类服务，请先运行 python src/05_predict_api.py，并等待服务启动。')
            except requests.exceptions.HTTPError as error:
                status = error.response.status_code if error.response is not None else '未知'
                st.error(f'分类服务返回错误（HTTP {status}），请查看接口运行窗口中的错误信息。')
            except ValueError:
                st.error('分类服务返回的数据格式异常，预期为包含 result.label 的 JSON。')
            except requests.exceptions.RequestException:
                st.error('请求分类服务失败，请检查服务状态后重试。')

    result = st.session_state.get('prediction')
    if result:
        st.subheader('分类结果')
        label = result['label']
        st.success(f"预测类别：{CLASS_NAMES.get(label, label)}")
        st.caption(f'类别标识：{label}')
        st.write('提交的新闻内容：')
        st.text(result['text'])


if __name__ == '__main__':
    main()
