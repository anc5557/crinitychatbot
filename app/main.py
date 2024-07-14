from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser


class Question(BaseModel):
    question: str


class ResponseModel(BaseModel):
    answer: str
    documents: List[str]


class DocumentResponse(BaseModel):
    documents: List[str]


app = FastAPI()

# 글로벌 변수로 모델 초기화
embedding_model = None
faiss_index = None
llm = None


def load_models():
    global embedding_model, faiss_index, llm

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask"
        )
        print("Embedding model loaded")

    if faiss_index is None:
        faiss_index = FAISS.load_local(
            folder_path="index/faiss_index",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        print("FAISS index loaded")

    if llm is None:
        llm = Ollama(model="EEVE-Korean-10.8B-Q5_K_M-GGUF")
        print("LLM model loaded")


@app.on_event("startup")
async def startup_event():
    load_models()


@app.post("/search/", response_model=DocumentResponse)
async def search(question: Question):
    result = faiss_index.similarity_search(question.question, k=1)
    documents = [doc.page_content for doc in result]
    return DocumentResponse(documents=documents)


@app.post("/generate/")
async def generate(question: Question):
    result = faiss_index.similarity_search(question.question, k=1)
    documents = [doc.page_content for doc in result]
    context = "\n\n".join(documents)

    system = """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
    검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
    한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. Don't narrate the answer, just answer the question. Let's think step-by-step."""

    human = """#Question: 
    {question} 

    #Context: 
    {context} 

    #Answer:"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | llm | StrOutputParser()

    def answer_streamer():
        for chunk in chain.stream({"question": question.question, "context": context}):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(answer_streamer(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
