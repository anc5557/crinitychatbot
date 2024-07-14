import streamlit as st
import requests
import json

# FastAPI 서버의 URL 설정
API_URL = "http://127.0.0.1:8000"


def search_question(question):
    """질문을 검색하고 관련 문서를 반환하는 함수"""
    response = requests.post(f"{API_URL}/search/", json={"question": question})
    if response.status_code == 200:
        return response.json()
    else:
        return {"documents": []}


def generate_answer_stream(question):
    """질문에 대한 답변을 스트리밍으로 생성하는 함수"""
    response = requests.post(
        f"{API_URL}/generate/", json={"question": question}, stream=True
    )
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    yield decoded_line[len("data: ") :]
    else:
        yield "오류가 발생했습니다."


# Streamlit 애플리케이션의 레이아웃을 구성
st.title("크리니티 Q&A")


question = st.text_input("질문을 입력해주세요:", key="question_input")

submit_button = st.button("답변 생성")

if submit_button or st.session_state.get("submitted"):
    with st.spinner("답변 중..."):
        answer_placeholder = st.empty()
        full_answer = ""
        key_count = 0

        for partial_answer in generate_answer_stream(st.session_state.question_input):
            full_answer += partial_answer
            key_count += 1
            answer_placeholder.text_area(
                "생성된 답변:",
                full_answer,
                height=300,
                key=f"generated_answer_{key_count}",
            )

        # 질문에 대한 문서 출력
        documents = search_question(st.session_state.question_input)["documents"]
        if documents:
            st.write("참고 문서:")
            for document in documents:
                st.write(document)
        else:
            st.write("관련 문서를 찾을 수 없습니다.")
