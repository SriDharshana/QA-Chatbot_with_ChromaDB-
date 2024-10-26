import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import chromadb
client = chromadb.Client()

collection_name = "text_collection" 
if collection_name in client.list_collections():
    client.delete_collection(collection_name)
collection = client.create_collection(name=collection_name) 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

folder_path = "/content/drive/MyDrive/dataset/" 
file_names = os.listdir(folder_path)
document_texts = []
document_ids = []


for idx, file_name in enumerate(file_names):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            text = file.read().strip()
            document_texts.append(text)
            document_ids.append(f"doc_{idx}")
            # Encode text and add it to ChromaDB
            embedding = embedding_model.encode([text])[0]
            collection.add(
                ids=[f"doc_{idx}"],
                embeddings=[embedding],
                documents=[text]
            )

def retrieve_passages(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_sentences = " ".join([doc for doc in results['documents'][0]])
    return retrieved_sentences

def generate_response(query):
    context = retrieve_passages(query)
    response = qa_model(question=query, context=context)
    return response['answer']


def gradio_chatbot(query):
    return generate_response(query)

def chatbot():
    print("Welcome to the RAG Chatbot!")
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        response = generate_response(query)
        print(f"Answer: {response}")

chatbot()
interface.launch()
