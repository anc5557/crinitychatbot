import streamlit as st
import requests

# FastAPI 서버의 URL 설정
API_URL = "http://127.0.0.1:8000"


def search_question(question):
    """질문을 검색하고 관련 문서를 반환하는 함수"""
    response = requests.post(f"{API_URL}/search/", json={"question": question})
    if response.status_code == 200:
        return response.json()
    else:
        return {"answer": "오류가 발생했습니다.", "documents": []}


def generate_answer(question):
    """질문에 대한 답변을 생성하는 함수"""
    response = requests.post(f"{API_URL}/generate/", json={"question": question})
    if response.status_code == 200:
        return response.json()
    else:
        return {"answer": "오류가 발생했습니다.", "documents": []}


# Streamlit 애플리케이션의 레이아웃을 구성
st.title("FAQ 챗봇 시스템")

question = st.text_input("질문을 입력해주세요:")

if st.button("검색"):
    with st.spinner("검색 중..."):
        search_results = search_question(question)
        st.text("검색된 답변:")
        st.write(search_results["answer"])
        st.text("관련 문서:")
        for doc in search_results["documents"]:
            st.write(doc)

if st.button("답변 생성"):
    with st.spinner("답변 생성 중..."):
        generated_results = generate_answer(question)
        st.text("생성된 답변:")
        st.write(generated_results["answer"])
        st.text("참고한 문서:")
        for doc in generated_results["documents"]:
            st.write(doc)
