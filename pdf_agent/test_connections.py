from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ✅ Load full PDF
loader = PyPDFLoader("./document/Resume.pdf")
documents = loader.load()

# ✅ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# ✅ Use Nomic Text Embedding (via Ollama)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ✅ Store embeddings in Chroma Vector DB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_db.persist()

print("✅ Embeddings created and stored successfully!")

query = "give me candidate email"
docs = vector_db.similarity_search(query, k=2)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
