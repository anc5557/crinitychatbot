from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate


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
        llm = HuggingFacePipeline.from_model_id(
            model_id="MLP-KTLim/llama-3-Korean-Bllossom-8B",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 20},
        )
        print("LLM model loaded")


@app.on_event("startup")
async def startup_event():
    load_models()


@app.post("/search/", response_model=DocumentResponse)
async def search(question: Question):
    result = faiss_index.similarity_search(question.question, k=2)
    print(f"result : \n\n {result}")
    documents = [doc.page_content for doc in result]
    print(f"documents : \n\n {documents}")
    return DocumentResponse(documents=documents)


@app.post("/generate/", response_model=ResponseModel)
async def generate(question: Question):
    result = faiss_index.similarity_search(question.question, k=2)
    documents = [doc.page_content for doc in result]

    combined_documents = "\n\n".join(documents)  # 두 개의 문서를 하나로 합침

    template = """
    질문: {question}
    참고 문서: {documents}
    답변: {answer}
    """

    prompt = PromptTemplate.from_template(template)

    # RunnableSequence를 사용하여 prompt와 llm을 연결
    llm_chain = RunnableSequence(prompt | llm)

    # 필요한 입력 키 'question'과 'documents'를 제공
    answer = llm_chain.invoke(
        {"question": question.question, "documents": combined_documents, "answer": ""}
    )

    return ResponseModel(
        answer=answer,
        documents=documents,  # 원본 문서를 리스트 형태로 유지
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
