import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)


st.title("OpenAI Chatbot")

# st.session_state : 세션에 키-값 형식으로 데이터를 저장하는 변수
# 저장하고 쓸 값들
# openai_model : gpt-4.1 or gpt-3.5-turbo
# messages     : [] 대화들
print('')
print('='*30)
print("Session State:")
print(st.session_state)
print('='*30)
print('')

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = "gpt-4.1" # or gpt-3.5-turbo

if 'messages' not in st.session_state:
    st.session_state.messages = []

# 기존의 메시지가 있다면 출력
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# prompt => 사용자 입력창
if prompt := st.chat_input("메시지를 입력하세요!") :
    # message => [], append, 대화 내용 추가
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model = st.session_state.openai_model,
            messages = [
                {"role" : m['role'], "content": m['content']}
                for m in st.session_state.messages
            ],
            stream = True
        )
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": response
        }
    )



