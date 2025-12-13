import streamlit as st
import os

# Deshabilitar TensorFlow para evitar problemas con Keras 3
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

# --- 1. Definición de Clases y Funciones ---

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

@st.cache_resource
def load_llm_model(model_id, model_type, _max_length, _temperature, _top_p, _repetition_penalty):
    """Carga el modelo LLM seleccionado con los parámetros configurados"""
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoTokenizer, pipeline
    
    st.info(f"Cargando LLM **{model_id}** localmente... (esto puede tardar un poco)")
    
    # Intentar cargar tokenizador, forzando versión lenta si es necesario
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    
    # Asegurar que existe un pad_token (crítico para GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cargar el modelo apropiado según el tipo
    if model_type == "seq2seq":
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        task = "text2text-generation"
    else:  # causal
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.pad_token_id = tokenizer.pad_token_id # Sincronizar pad_token
        task = "text-generation"
    
    # Configurar argumentos del pipeline
    pipeline_kwargs = {
        "task": task,
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": _max_length,
        "temperature": _temperature,
        "top_p": _top_p,
        "repetition_penalty": _repetition_penalty,
    }

    # Solo agregar return_full_text para modelos causales (GPT-2, etc)
    # Los modelos seq2seq (FLAN-T5) no soportan este parámetro y fallan si se pasa
    if model_type == "causal":
        pipeline_kwargs["return_full_text"] = False

    # Crear pipeline con parámetros configurables
    pipe = pipeline(**pipeline_kwargs)
    
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_and_index_documents(_index_name, _embeddings, _pinecone_client):
    """
    Carga los documentos, realiza el chunking y los indexa en Pinecone.
    """
    st.info("Cargando y procesando documentos...")
    
    # Simulación de carga de un documento
    try:
        loader = TextLoader("cv_alumno.txt") 
        documents = loader.load()
    except FileNotFoundError:
        st.error("Archivo 'cv_alumno.txt' no encontrado. Por favor, créalo.")
        return None

    # Chunking (División)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    
    st.success(f"Documentos divididos en {len(docs)} chunks.")

    # Verificar si el índice existe y crearlo si no
    import time
    try:
        existing_indexes = [i.name for i in _pinecone_client.list_indexes()]
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            st.error("⛔ **Error de Autenticación en Pinecone**")
            st.error("Tu API Key de Pinecone parece ser inválida. Por favor verifícala en la configuración lateral.")
            st.stop()
        else:
            st.error(f"Error al conectar con Pinecone: {e}")
            return None

    if _index_name not in existing_indexes:
        st.info(f"Creando índice '{_index_name}' en Pinecone...")
        try:
            _pinecone_client.create_index(
                name=_index_name,
                dimension=384,  # Dimensión de paraphrase-multilingual-MiniLM-L12-v2
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            st.info("Esperando a que el índice esté listo...")
            time.sleep(10) # Esperar a que Pinecone inicialice
            st.success(f"Índice '{_index_name}' creado exitosamente.")
        except Exception as e:
            if "max serverless indexes allowed" in str(e):
                st.error("⛔ **Error: Límite de índices alcanzado en Pinecone**")
                try: 
                    st.error("Has alcanzado el máximo de 5 índices permitidos en el plan gratuito.")
                except:
                   pass
                st.info("💡 **Solución**: Ve a [app.pinecone.io](https://app.pinecone.io) y elimina algún índice que no uses.")
                st.stop()
            else:
                st.error(f"Error al crear índice: {e}")
                st.stop()
        
        # Carga masiva de embeddings
        st.info(f"Subiendo {len(docs)} chunks a Pinecone...")
        vectorstore = PineconeVectorStore.from_documents(
            docs, 
            _embeddings, 
            index_name=_index_name
        )
        st.success("Embeddings cargados a Pinecone.")
        return vectorstore
    else:
        # Si el índice ya existe, solo conectamos
        st.info(f"Conectando al índice existente: '{_index_name}'.")
        vectorstore = PineconeVectorStore(index_name=_index_name, embedding=_embeddings)
        
        # --- VERIFICACIÓN DE ESTADO ---
        try:
            index = _pinecone_client.Index(_index_name)
            stats = index.describe_index_stats()
            count = stats.total_vector_count
            st.write(f"📊 Estado del índice: {count} vectores totales.")
            if count == 0:
                 st.warning("⚠️ El índice está vacío. Intentando re-indexar...")
                 st.info(f"Subiendo {len(docs)} chunks a Pinecone...")
                 vectorstore.add_documents(docs)
                 st.success("Embeddings cargados.")
        except Exception as e:
            st.warning(f"No se pudo verificar el estado del índice: {e}")
            
        return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 2. Interfaz de Usuario y Configuración ---

st.set_page_config(page_title="RAG Chatbot con Pinecone 🤖", layout="wide")
st.title("🤖 RAG Chatbot con Pinecone (Gratuito) y modelos locales")

# Sidebar
pinecone_api_key = st.sidebar.text_input("Pinecone API Key:", type="password")
pinecone_env = st.sidebar.text_input("Pinecone Environment (Ej: gcp-starter):")
index_name = st.sidebar.text_input("Nombre del Índice de Pinecone:", "my-rag-index")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Configuración del Modelo")

model_options = {
    "FLAN-T5 Base (Recomendado)": {
        "model_id": "google/flan-t5-base",
        "type": "seq2seq",
        "description": "Modelo equilibrado, bueno para español"
    },
    "FLAN-T5 Small (Rápido)": {
        "model_id": "google/flan-t5-small",
        "type": "seq2seq",
        "description": "Más rápido pero menos preciso"
    },
    "FLAN-T5 Large (Mejor calidad)": {
        "model_id": "google/flan-t5-large",
        "type": "seq2seq",
        "description": "Mejor calidad pero más lento"
    }
}

selected_model_name = st.sidebar.selectbox(
    "Selecciona el modelo LLM:",
    options=list(model_options.keys()),
    index=0
)

selected_model_info = model_options[selected_model_name]
st.sidebar.info(f"📝 {selected_model_info['description']}")

st.sidebar.markdown("### ⚙️ Parámetros de Generación")
max_length = st.sidebar.slider("Longitud máxima:", 64, 512, 256, 32)
temperature = st.sidebar.slider("Temperature:", 0.01, 1.0, 0.1, 0.01)
top_p = st.sidebar.slider("Top P:", 0.1, 1.0, 0.95, 0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty:", 1.0, 2.0, 1.1, 0.05)

if st.sidebar.button("🔄 Limpiar Caché y Recargar Modelo"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.info("ℹ️ **Nota**: Este chatbot usa embeddings de HuggingFace (dimensión 384).")

if not all([pinecone_api_key, pinecone_env, index_name]):
    st.warning("Por favor, introduce la API key de Pinecone y configuraciones en la barra lateral.")
    st.stop()

# --- 3. Inicialización del Sistema ---

try:
    # 1. Pinecone
    pinecone_client = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    
    # 2. Embeddings
    st.info("Cargando modelo de embeddings... (puede tardar unos minutos la primera vez)")
    embeddings = CustomHuggingFaceEmbeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    st.success("✅ Modelo de embeddings cargado correctamente")
    
    # 3. LLM
    llm = load_llm_model(
        selected_model_info["model_id"],
        selected_model_info["type"],
        max_length,
        temperature,
        top_p,
        repetition_penalty
    )
    st.success(f"✅ LLM **{selected_model_name}** cargado correctamente")

except Exception as e:
    st.error(f"Error al inicializar modelos: {e}")
    st.stop()

# --- 4. Carga e Indexación de Documentos ---

vectorstore = load_and_index_documents(index_name, embeddings, pinecone_client)

if vectorstore:
    retriever = vectorstore.as_retriever()
else:
    st.error("No se pudo iniciar el servicio de recuperación.")
    st.stop()

# --- 5. Lógica del Chatbot ---

st.subheader("Test de Recuperación")
test_query = "Experiencia Ezequiel Caamano"
if st.button(f"Probar Recuperación: **{test_query}**"):
    docs = retriever.invoke(test_query)
    if docs:
        st.success("Recuperación exitosa:")
        for i, d in enumerate(docs):
            st.code(f"Chunk {i+1}: {d.page_content[:200]}...")
    else:
        st.warning("No se encontraron documentos.")

st.markdown("---")
st.subheader("Chat con el CV 🤖")

# Prompt Template
if selected_model_info["type"] == "seq2seq":
    template = """Responde la pregunta basándote estrictamente en el contexto proporcionado.

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""
else:
    template = """Contexto con información relevante:
{context}

Pregunta del usuario:
{question}

Respuesta del asistente:"""

RAG_PROMPT = PromptTemplate.from_template(template)

# Cadena RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hazme una pregunta sobre el CV..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Pensando..."):
        try:
            response = rag_chain.invoke(prompt)
            # source_docs = retriever.invoke(prompt) # Descomentar si quieres ver fuentes siempre
            
            with st.chat_message("assistant"):
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Error generando respuesta: {e}")
