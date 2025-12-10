from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def load_chroma():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_db = Chroma(
        persist_directory="../chroma_db",
        embedding_function=embeddings
    )
    
    return vector_db
