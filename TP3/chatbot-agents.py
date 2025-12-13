import streamlit as st
import os
from typing import List

# ------------------------------
# Configuración de entorno
# ------------------------------
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Imports
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# ------------------------------
# Clases y Funciones Auxiliares
# ------------------------------
class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

def load_and_index_agent(_agent_key, _agent_data, _embeddings, _pinecone_client):
    index_name = _agent_data["index"]
    file_path = _agent_data["file"] # Ruta corregida
    
    if not os.path.exists(file_path):
        st.error(f"❌ Archivo no encontrado para {_agent_data['name']}: {file_path}")
        return None

    try:
        existing_indexes = [i.name for i in _pinecone_client.list_indexes()]
    except Exception as e:
        st.error(f"Error conectando a Pinecone: {e}")
        return None

    if index_name not in existing_indexes:
        with st.spinner(f"Creando índice para {_agent_data['name']}..."):
            try:
                _pinecone_client.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                import time
                time.sleep(10)
            except Exception as e:
                if "max serverless indexes" in str(e):
                    st.error("⛔ Max indexes reached.")
                    st.stop()
                raise e
            
            loader = TextLoader(file_path)
            docs = loader.load()
            # Chunking optimizado: 600 chars aseguran que en 3-4 chunks entre todo el CV
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            vectorstore = PineconeVectorStore.from_documents(
                chunks, _embeddings, index_name=index_name
            )
            st.toast(f"✅ Agente {_agent_data['name']} indexado correctamente.")
            return vectorstore
    else:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=_embeddings)
        index = _pinecone_client.Index(index_name)
        if index.describe_index_stats().total_vector_count == 0:
             st.warning(f"Índice de {_agent_data['name']} vacío. Re-indexando...")
             loader = TextLoader(file_path)
             splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
             chunks = splitter.split_documents(loader.load())
             vectorstore.add_documents(chunks)
        
        return vectorstore

@st.cache_resource
def load_llm_model():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        repetition_penalty=1.2,
    )
    return HuggingFacePipeline(pipeline=pipe)

# ------------------------------
# Definición de Agentes
# ------------------------------
AGENTS_CONFIG = {
    "alumno": {
        "name": "Alumno (Ezequiel)",
        "file": "alumno.txt", 
        "index": "agent-alumno",
        "aliases": ["alumno", "ezequiel", "mi", "yo"]
    },
    "mariela": {
        "name": "Mariela Rodríguez",
        "file": "mariela.txt",
        "index": "agent-mariela",
        "aliases": ["mariela", "rodríguez", "gabriela"]
    },
    "juan": {
        "name": "Juan Pérez",
        "file": "juan.txt",
        "index": "agent-juan",
        "aliases": ["juan", "pérez"]
    },
    "carlos": {
        "name": "Carlos Gómez",
        "file": "carlos.txt",
        "index": "agent-carlos",
        "aliases": ["carlos", "gómez"]
    }
}

# ------------------------------
# Interfaz Gráfica
# ------------------------------
st.set_page_config(page_title="Sistema Multi-Agente RAG", layout="wide")
st.title("🤖 Chatbot Multi-Agente de RRHH")
st.markdown("Consulta información de múltiples perfiles simultáneamente.")

# Sidebar
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
pinecone_env = st.sidebar.text_input("Pinecone Environment")

if not pinecone_api_key or not pinecone_env:
    st.info("👈 Por favor configura tus credenciales de Pinecone para comenzar.")
    st.stop()

# ------------------------------
# Inicialización
# ------------------------------
try:
    pinecone_client = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    embeddings = CustomHuggingFaceEmbeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    llm = load_llm_model()
except Exception as e:
    st.error(f"Error de inicialización: {e}")
    st.stop()

# Cargar Agentes
VECTORSTORES = {}
st.sidebar.markdown("---")
st.sidebar.markdown("### 🟢 Estado de Agentes")

for key, config in AGENTS_CONFIG.items():
    vs = load_and_index_agent(key, config, embeddings, pinecone_client)
    if vs:
        VECTORSTORES[key] = vs
        st.sidebar.success(f"{config['name']} Online")
    else:
        st.sidebar.error(f"{config['name']} Offline")

# ------------------------------
# Lógica de Ruteo y Chat
# ------------------------------
def get_relevant_context(query):
    query_lower = query.lower()
    targeted_agents = []
    
    # 1. Detectar agentes mencionados
    for key, config in AGENTS_CONFIG.items():
        for alias in config["aliases"]:
            if alias in query_lower:
                targeted_agents.append(key)
                break
    
    # 2. Fallback por defecto ("alumno")
    if not targeted_agents:
        targeted_agents = ["alumno"]
        
    st.toast(f"Consultando agentes: {', '.join(targeted_agents).upper()}")
        
    # 3. Recuperar contexto de cada agente
    full_context = ""
    
    # ESTRATEGIA ADAPTATIVA (Clave para FLAN-T5)
    # Single Agent -> k=4 (Trae casi todo el CV, incluyendo Idiomas al final)
    # Multi Agent -> k=2 (Trae lo más relevante de c/u para que ambos entren en contexto)
    k_val = 4 if len(targeted_agents) == 1 else 2
    
    for agent_key in targeted_agents:
        if agent_key in VECTORSTORES:
            retriever = VECTORSTORES[agent_key].as_retriever(search_kwargs={"k": k_val})
            docs = retriever.invoke(query)
            
            agent_name = AGENTS_CONFIG[agent_key]["name"].upper()
            
            # Deduplicación stricta
            seen_content = set()
            unique_docs = []
            for d in docs:
                content_clean = d.page_content.strip()
                if content_clean not in seen_content:
                    seen_content.add(content_clean)
                    unique_docs.append(content_clean)
            
            context_text = "\n".join(unique_docs)
            
            # Separador muy explícito para evitar alucinaciones cruzadas
            full_context += f"\n=== PERFIL DE {agent_name} ===\n{context_text}\n"
            
    return full_context

# Prompt ajustado
prompt_template = """Use the provided context to answer the question. 
If comparing, list key points for each person. 
If asked about specific details (like languages or skills), list them explicitly.

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate.from_template(prompt_template)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ej: ¿Qué experiencia tiene Mariela?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Consultando a los agentes..."):
        context = get_relevant_context(user_query)
        final_prompt = PROMPT.format(context=context, question=user_query)
        response = llm.invoke(final_prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
