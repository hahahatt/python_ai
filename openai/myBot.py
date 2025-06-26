import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import datetime

# 환경 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

st.title("🧠 OpenAI Chatbot")

# 세션 상태 초기화
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-4.1"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "active_chat_title" not in st.session_state:
    st.session_state.active_chat_title = None

# 사이드바
# 채팅 목록
with st.sidebar:
    st.markdown("## 💬 채팅 목록")

    if st.button("🆕 새 채팅 시작"):
        st.session_state.messages = []
        st.session_state.active_chat_title = None

    for title in list(st.session_state.chat_history.keys())[::-1]:
        if st.button(f"💬 {title}", key=title):
            st.session_state.messages = st.session_state.chat_history[title].copy()
            st.session_state.active_chat_title = title

    st.markdown("---")
    st.caption("첫 질문 후 자동으로 제목이 생성됩니다.")

# 기존 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 채팅 입력 처리
if prompt := st.chat_input("메시지를 입력하세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # 대화 제목 생성 & 저장
    if st.session_state.active_chat_title is None:
        first_prompt = st.session_state.messages[0]["content"]

        summary_response = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[
                {"role": "system", "content": "다음 문장을 채팅 제목으로 20자 이내로 요약해줘."},
                {"role": "user", "content": first_prompt}
            ]
        )
        title = summary_response.choices[0].message.content.strip()

        st.session_state.chat_history[title] = st.session_state.messages.copy()
        st.session_state.active_chat_title = title
    else:
        # 기존 대화라면 해당 제목 아래에 계속 덮어쓰기 저장
        st.session_state.chat_history[st.session_state.active_chat_title] = st.session_state.messages.copy()