import os, pathlib
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 

docs_path = pathlib.Path("documentos")
all_chunks = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100, 
    separators=["\n\n", "\n", " ", ""]
)

for file in docs_path.glob("*"):
    if file.suffix.lower() in [".pdf",".txt",".docx", ".doc"]:
        elements = partition(filename=str(file))
        text = "\n".join(e.text for e in elements if e.text)
        chunks = splitter.create_documents([text], metadatas=[{"source": file.name}])
        print(f"Leido {file.name} y creado {len(chunks)} fragmentos.")
        all_chunks.extend(chunks)

print("Total de fragmentos de texto:", len(all_chunks))

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(all_chunks, emb)
db.save_local("db")
print("Índice FAISS creado con", len(all_chunks), "trozos de texto.")