import streamlit as st
import requests
# Set the title of the app
st.title("新闻分类预测")

text_input = st.text_area('请输入需要预测的文本','郑州马拉松今日正式开跑')
if st.button('预测'):
    data = {'text':text_input}
    url = 'http://127.0.0.1:8000/predict'
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            st.write('预测结果:', result)
        else:
            st.write('请求失败:', response.status_code)
    except requests.exceptions.RequestException as e:
        st.write('请求异常:', e)
