from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import sys
import json


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_title_description(manuals):
    combined_docs = []
    for manual in manuals:
        for item in manual["manual"]:
            combined_docs.append(
                {"title": item["title"], "description": item["description"]}
            )
    return combined_docs


def main(json_filepath):
    # Load the JSON data
    data = load_json(json_filepath)

    # Combine title and description
    documents = []
    for entry in data:
        documents.extend(combine_title_description([entry]))

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # Create embeddings
    texts = [doc["title"] + " " + doc["description"] for doc in documents]

    # Convert texts to Document objects
    document_objects = [Document(page_content=text) for text in texts]

    # Create FAISS index
    db = FAISS.from_documents(documents=document_objects, embedding=embedding_model)

    # Save the FAISS index
    db.save_local("./index/faiss_index")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_vector.py <json_filepath>")
        sys.exit(1)

    json_filepath = sys.argv[1]
    main(json_filepath)
