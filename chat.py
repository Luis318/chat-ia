from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM         
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1) Cargar embeddings e índice
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "db",
    embeddings,
    allow_dangerous_deserialization=True        # quita esto si más adelante guardas en formato seguro
)
retriever = db.as_retriever(search_kwargs={"k": 3})

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
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False,
)

print("Chatbot académico listo. Escribe 'salir' para terminar.\n")

# 5) Bucle de conversación
while True:
    pregunta = input("Tú: ")
    if pregunta.lower() in {"salir", "exit"}:
        break

    # invoke() es la API recomendada; devuelve un dict
    respuesta = qa.invoke({"query": pregunta})["result"]
    print("Bot:", respuesta, "\n")
