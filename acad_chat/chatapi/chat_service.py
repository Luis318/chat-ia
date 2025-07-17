from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM         
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

# 1) Cargar embeddings e índice
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "db" 

try:
    db = FAISS.load_local(
        "db",
        embeddings,
        allow_dangerous_deserialization=True        # quita esto si más adelante guardas en formato seguro
    )
except (FileNotFoundError, RuntimeError):
    logging.error(
        f"No se encontro el inidice FAISS en {db_path}. "
        "Ejecuta `python ingest.py` para generarlo"
    )
    db = None

retriever = db.as_retriever(search_kwargs={"k": 3}) if db else None

# 2) Instanciar el modelo de Ollama
llm = OllamaLLM(model="llama3")                 # asegúrate de haber hecho: ollama pull llama3

# 3) Prompt con indicación de “no sé”
prompt = PromptTemplate(
    template=(
        "Eres un asistente académico de la universidad. "
        "Contesta SOLO con la información dada.\n\n"
        "{context}\n\n"
        "Pregunta: {question}\n"
        "Respuesta (si no está en el contexto di claramente "
        "que no tienes la información):"
    ),
    input_variables=["context", "question"],
)
# 4) Cadena QA
qa = (
    RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    if retriever 
    else None
)

def get_answer(question: str) -> str:
    """Devuelve la respuesta generada por el LLM"""
    if qa is None:
        return "El índice de conocimiento no está disponible. Por favor, ejecuta `ingest.py` para generarlo."
    return qa.invoke({"query": question})["result"]