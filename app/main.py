from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


app = FastAPI()


class question(BaseModel):
    question: str


class Response(BaseModel):
    answer: str
    documents: List[str]


class DocumentResponse(BaseModel):
    documents: List[str]


embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
print("Embedding model loaded")

# FAISS 인덱스 로드
faiss_index = FAISS.load_local(
    folder_path="index/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)
print("FAISS index loaded")

llm = HuggingFacePipeline.from_model_id(
    model_id="MLP-KTLim/llama-3-Korean-Bllossom-8B",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)
print("LLM model loaded")


@app.post("/search/", response_model=Response)
async def search(question: question):
    # FAISS 인덱스에서 가장 유사한 문서를 찾아 반환
    question_embedding = embedding_model.embed_query(question.question)
    result = faiss_index.similarity_search(question_embedding, k=1)
    document = result[0][0]
    return Response(
        answer=document.page_content,
        documents=[doc.page_content for doc in result[0]],
    )


@app.post("/generate/", response_model=Response)
async def generate(question: question):
    # 문서 검색
    question_embedding = embedding_model.embed_query(question.question)
    result = faiss_index.similarity_search(question_embedding, k=1)
    document = result[0][0]

    template = """
    질문: {question.question}
    참고 문서: {document.page_content}
    답변: {answer}
    """

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # LLMChain 호출 수정
    answer = llm_chain.invoke(
        {"question": question.question, "document": document.page_content}
    )  # 수정된 부분

    return Response(
        answer=answer,
        documents=[doc.page_content for doc in result[0]],
    )
