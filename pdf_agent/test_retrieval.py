from langchain_community.vectorstores import Chroma 
# ✅ Store embeddings in Chroma Vector DB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
query = "What is the candidate's experience?"
docs = vector_db.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
