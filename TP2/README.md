# Documentación: Sistema RAG Chatbot con Pinecone

## Índice
1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Tecnologías Utilizadas](#tecnologías-utilizadas)
4. [Decisiones de Diseño](#decisiones-de-diseño)
5. [Problemas Encontrados y Soluciones](#problemas-encontrados-y-soluciones)
6. [Funcionamiento del Sistema](#funcionamiento-del-sistema)
7. [Guía de Uso](#guía-de-uso)
8. [Limitaciones y Hardware](#limitaciones-y-hardware)
9. [Conclusiones](#conclusiones)

---

## Introducción

Este proyecto implementa un sistema de **Retrieval-Augmented Generation (RAG)** para crear un chatbot capaz de responder preguntas sobre un CV. El sistema combina técnicas de recuperación de información con modelos de lenguaje para generar respuestas contextualizadas y precisas.

### Objetivos del Trabajo Práctico
- Cargar un documento (CV) y generar embeddings vectoriales
- Almacenar los vectores en Pinecone (base de datos vectorial)
- Implementar búsqueda por similitud coseno
- Crear un chatbot interactivo usando Streamlit

---

## Arquitectura del Sistema

El sistema sigue el patrón RAG clásico:

```
Usuario → Pregunta → Embedding → Búsqueda en Pinecone 
                                         ↓
                                  Recuperación de contexto
                                         ↓
                            Prompt + Contexto + Pregunta
                                         ↓
                                    Modelo LLM
                                         ↓
                                    Respuesta
```

### Componentes Principales

1. **Embeddings**: Conversión de texto a vectores numéricos
2. **Vector Store (Pinecone)**: Almacenamiento y búsqueda de vectores
3. **Retriever**: Sistema de recuperación por similitud
4. **LLM**: Modelo de lenguaje para generación de respuestas
5. **Interfaz (Streamlit)**: UI para interacción con el usuario

---

## Tecnologías Utilizadas

### Librerías Core
- **Streamlit**: Framework para la interfaz web
- **LangChain**: Orquestación del pipeline RAG
- **Pinecone**: Base de datos vectorial en la nube
- **Sentence-Transformers**: Generación de embeddings
- **Transformers (HuggingFace)**: Modelos de lenguaje

### Modelos

#### Embeddings
- **paraphrase-multilingual-MiniLM-L12-v2**
  - Dimensión: 384
  - Soporte multilingüe (español/inglés)
  - Optimizado para similitud semántica

#### LLMs Disponibles
1. **FLAN-T5 (Recomendado)**
   - Variantes: Small, Base, Large
   - Tipo: Seq2Seq (Text-to-Text)
   - Entrenado con instrucciones multilingües

2. **GPT-2** (Problemático - ver sección de problemas)
3. **mT5** (Problemático - ver sección de problemas)

---

## Evolución del Proyecto y Decisiones de Diseño

### Intentos Iniciales con HuggingFace Inference API

En la primera etapa del proyecto, se intentó utilizar modelos de HuggingFace a través de su **Inference API** usando `HuggingFaceEndpoint`:

```python
# Intento inicial (NO FUNCIONÓ)
from langchain_huggingface import HuggingFaceEndpoint

# Se intentaron varios modelos vía API:
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=api_token
)

# También se probó Mistral
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=api_token
)
```

#### Modelos Intentados vía API (Todos Fallaron)

1. **FLAN-T5 Base** - Timeouts constantes
2. **Mistral-7B-Instruct-v0.2** - Rate limits excedidos, tiempos de espera prohibitivos
3. **GPT-2** - Respuestas lentas e inconsistentes

#### Problemas Encontrados

1. **Límites de Rate (Requests por minuto)**: La API gratuita tiene restricciones severas
   - Plan gratuito: ~1000 requests/día
   - Mistral-7B: Especialmente problemático por ser muy popular
2. **Timeouts frecuentes**: Los modelos tardan en "despertar" si no están en memoria
   - Mistral-7B: Hasta 2-3 minutos de cold start
   - Timeouts de 30 segundos hacían imposible su uso
3. **Dependencia de conexión**: No funciona offline
4. **Cuotas limitadas**: El plan gratuito se agota rápidamente
5. **Colas de espera**: Modelos grandes como Mistral tienen prioridad baja en plan gratuito

#### Solución Adoptada: Modelos Locales

Se decidió cambiar a **modelos locales** usando `HuggingFacePipeline`:

```python
from transformers import AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Cargar modelo localmente
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)
```

**¿Por qué no se intentó Mistral-7B local?**
- **Tamaño prohibitivo**: ~15 GB de modelo
- **Requisitos de RAM**: 16+ GB solo para el modelo
- **Sin GPU**: Generación extremadamente lenta (minutos por respuesta)
- **Incompatible con el objetivo**: Sistema gratuito accesible en hardware modesto

**Ventajas de modelos locales más pequeños (FLAN-T5)**:
- ✅ Sin límites de requests
- ✅ Funciona offline
- ✅ Latencia predecible
- ✅ Control total sobre parámetros
- ✅ Tamaño manejable (1-3 GB)

**Desventajas**:
- ❌ Requiere 2-6 GB RAM según modelo
- ❌ Primera carga lenta (descarga del modelo)
- ❌ Sin GPU, generación más lenta que API (pero más estable)

### Eliminación de GPT-2 y mT5

Durante el desarrollo se probaron múltiples modelos, pero **GPT-2 y mT5 fueron eliminados de la versión final** por no responder adecuadamente a las necesidades del sistema RAG.

#### Modelos Descartados

```python
# CÓDIGO ELIMINADO - Estos modelos NO están en la versión final

# GPT-2 - DESCARTADO
"GPT-2 Medium": {
    "model_id": "gpt2-medium",
    "type": "causal"
}

# mT5 - DESCARTADO  
"mT5 Base": {
    "model_id": "google/mt5-base",
    "type": "seq2seq"
}
```

#### Razones de Eliminación

**GPT-2 fue descartado porque**:
- Generaba respuestas irrelevantes o inventadas
- No seguía las instrucciones del prompt
- Tendía a continuar el texto en lugar de responder
- Calidad inaceptable para un sistema de Q&A

**mT5 fue descartado porque**:
- Respuestas genéricas o vacías
- Requería formato de prompt muy específico
- Sin instruction-tuning, no comprende la tarea
- Necesitaría fine-tuning adicional (fuera del alcance)

#### Versión Final: Solo FLAN-T5

La versión final del código **únicamente incluye variantes de FLAN-T5**:

```python
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
```

Estos modelos demostraron ser los **únicos viables** para el sistema RAG sin necesidad de fine-tuning adicional.

---

## Decisiones de Diseño

### 1. Modelo de Embeddings

**Decisión**: Usar `paraphrase-multilingual-MiniLM-L12-v2`

**Justificación**:
- Soporte nativo para español
- Balance entre calidad y velocidad
- Dimensionalidad manejable (384)
- Optimizado para tareas de similitud semántica

### 2. Chunking Strategy

```python
chunk_size=200
chunk_overlap=50
separators=["\n\n", "\n", ".", "!", "?", " ", ""]
```

**Justificación**:
- **200 caracteres**: Mantiene contexto coherente sin ser excesivo
- **50 de overlap**: Evita perder información en los límites
- **Separadores jerárquicos**: Respeta estructura natural del texto

### 3. Selección de FLAN-T5 como Modelo Principal

**Ventajas sobre GPT-2/mT5**:
- Arquitectura Seq2Seq diseñada para seguir instrucciones
- Entrenamiento con "instruction tuning"
- Mejor comprensión de contexto estructurado
- Soporte multilingüe nativo

### 4. Sistema de Caché

```python
@st.cache_resource
def load_llm_model(...):
    ...
```

**Justificación**:
- Los modelos son pesados (>500MB)
- Evita recargas innecesarias
- Mejora experiencia del usuario

### 5. Validación de Pinecone

El sistema incluye:
- Verificación de autenticación
- Manejo de límites del plan gratuito (5 índices)
- Reconexión a índices existentes
- Verificación de estado (vectores cargados)

---

## Problemas Encontrados y Soluciones

### 🔴 Problema 1: Modelos de HuggingFace Inference API No Cargaban

#### Síntoma Inicial

Al intentar usar `HuggingFaceEndpoint` para acceder a modelos remotos:

```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=token
)
```

**Errores encontrados**:
- `TimeoutError`: El modelo tarda demasiado en responder
- `RateLimitError`: Límite de requests excedido
- `503 Service Unavailable`: Modelo no disponible
- Respuestas inconsistentes y lentas

#### Causa Raíz

**Limitaciones de la Inference API gratuita**:
1. **Cold start**: Si el modelo no está en memoria, tarda minutos en cargar
2. **Rate limits**: ~1000 requests/día en plan gratuito
3. **Colas de espera**: Usuarios comparten recursos
4. **Timeouts agresivos**: 30 segundos máximo por request

#### Solución: Migración a Modelos Locales

Se cambió completamente a `HuggingFacePipeline` con modelos descargados:

```python
from transformers import AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)
```

**Resultado**: Sistema estable, predecible y sin límites de uso.

---

### 🔴 Problema 2: GPT-2 Genera Respuestas Irrelevantes (MODELO ELIMINADO)

> **⚠️ IMPORTANTE**: GPT-2 fue **completamente eliminado** de la versión final por generar respuestas inadecuadas.

#### ¿Por qué GPT-2 fue descartado?

**1. Arquitectura Causal (Autoregresivo)**

GPT-2 está diseñado para **continuar texto**, no para responder preguntas:

```
Input: "El CV dice que trabajó en X. ¿Dónde trabajó?"
GPT-2 piensa: "Debo continuar esta historia..."
Output: "Trabajó en varios lugares interesantes y además..."
```

**2. Entrenamiento sin Instrucciones**

- GPT-2 fue entrenado con texto de internet sin formato de Q&A
- No comprende la estructura "Contexto → Pregunta → Respuesta"
- Tiende a generar texto creativo en lugar de respuestas factuales

**3. Problema del pad_token**

```python
# GPT-2 no tiene pad_token por defecto
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Sin esto, el modelo no puede procesar correctamente batches.

**4. Fuga de Contexto**

GPT-2 no diferencia bien entre el contexto recuperado y la pregunta, mezclando ambos en su generación.

#### Ejemplos de Respuestas Problemáticas

```python
# Pregunta: "¿Cuál es la experiencia laboral de María?"
# Respuesta GPT-2: "La experiencia laboral de María es muy interesante 
# y variada, incluyendo múltiples proyectos en diferentes áreas..."
# ❌ Respuesta genérica sin datos reales del CV

# Pregunta: "¿Dónde estudió?"
# Respuesta GPT-2: "Estudió en una prestigiosa universidad donde..."
# ❌ Inventando información no presente en el contexto
```

#### Decisión: Eliminación Completa

**GPT-2 NO está en la versión final del código**. Fue reemplazado exclusivamente por FLAN-T5.

---

### 🔴 Problema 3: mT5 No Funcionaba Bien (MODELO ELIMINADO)

> **⚠️ IMPORTANTE**: mT5 fue **completamente eliminado** de la versión final por respuestas inadecuadas.

#### ¿Por qué mT5 fue descartado?

**1. Entrenamiento No Supervisado**

- mT5 base fue pre-entrenado solo con tareas de "span corruption"
- No fue entrenado específicamente para Q&A
- Necesita fine-tuning para tareas específicas

**2. Formato de Prompt Incorrecto**

mT5 espera prefijos específicos:

```python
# Formato correcto para mT5
prompt = "question: ¿Dónde estudió? context: María estudió en..."

# Formato usado (incorrecto para mT5 base)
prompt = "Contexto:\n{context}\n\nPregunta:\n{question}"
```

**3. Tokenización Problemática**

mT5 usa SentencePiece, que puede tener problemas con formato de prompts no estándar.

#### Ejemplos de Respuestas Problemáticas

```python

# Pregunta: "Describe la experiencia laboral"
# Respuesta mT5: ""
# ❌ Respuesta vacía
```

#### Decisión: Eliminación Completa

**mT5 NO está en la versión final del código**. Fue reemplazado exclusivamente por FLAN-T5.

---

### 🟢 Solución Final: FLAN-T5 (ÚNICO MODELO EN VERSIÓN FINAL)

> **✅ La versión final del código SOLO incluye variantes de FLAN-T5**. GPT-2 y mT5 fueron completamente eliminados.

#### ¿Por qué FLAN-T5 funciona mejor?

**1. Instruction Tuning**

FLAN-T5 fue entrenado específicamente con instrucciones:

```
Task: Responde basándote en el contexto
Input: [contexto] Pregunta: [pregunta]
Output: [respuesta]
```

**2. Arquitectura Seq2Seq Apropiada**

- **Encoder**: Procesa contexto + pregunta juntos
- **Decoder**: Genera respuesta de forma independiente
- No intenta "continuar" el texto como GPT-2

**3. Entrenamiento Multilingüe con Instrucciones**

- Datasets de Q&A en múltiples idiomas
- Comprende la tarea sin necesitar fine-tuning adicional

**4. No Necesita return_full_text**

```python
if model_type == "causal":
    pipeline_kwargs["return_full_text"] = False
# FLAN-T5 (seq2seq) NO necesita esto
```

---

### 🔴 Problema 4: Límites de Pinecone (Plan Gratuito)

#### Síntoma
```
Error: max serverless indexes allowed
```

#### Causa
- Plan gratuito: máximo 5 índices
- Cada prueba creaba un índice nuevo

#### Solución Implementada

```python
if "max serverless indexes allowed" in str(e):
    st.error("⛔ Límite de índices alcanzado")
    st.info("💡 Ve a app.pinecone.io y elimina índices")
    st.stop()
```

Además, el sistema verifica si el índice ya existe antes de crear uno nuevo.

---

### 🔴 Problema 5: Índices Vacíos

#### Síntoma
El sistema se conecta al índice pero no devuelve resultados.

#### Causa
- Índice creado pero vectores no cargados
- Proceso de carga interrumpido

#### Solución

```python
stats = index.describe_index_stats()
if stats.total_vector_count == 0:
    st.warning("⚠️ Índice vacío. Re-indexando...")
    vectorstore.add_documents(docs)
```

---

### 🔴 Problema 6: Autenticación de Pinecone

#### Solución

```python
try:
    existing_indexes = pinecone_client.list_indexes()
except Exception as e:
    if "401" in str(e) or "Unauthorized" in str(e):
        st.error("⛔ API Key inválida")
        st.stop()
```

Validación temprana evita errores crípticos más adelante.

---

## Funcionamiento del Sistema

### Fase 1: Inicialización

```python
# 1. Cargar modelo de embeddings
embeddings = CustomHuggingFaceEmbeddings(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Conectar a Pinecone
pinecone_client = Pinecone(api_key=..., environment=...)

# 3. Cargar LLM
llm = load_llm_model(model_id, model_type, ...)
```

### Fase 2: Procesamiento de Documentos

```python
# 1. Cargar CV
loader = TextLoader("cv_mariela.txt")
documents = loader.load()

# 2. Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Generar embeddings y subir a Pinecone
vectorstore = PineconeVectorStore.from_documents(
    docs, embeddings, index_name=index_name
)
```

### Fase 3: Pipeline RAG

```python
# 1. Usuario hace pregunta
prompt = "¿Dónde estudió María?"

# 2. Convertir pregunta a embedding
query_embedding = embeddings.embed_query(prompt)

# 3. Buscar en Pinecone (similitud coseno)
relevant_docs = retriever.invoke(prompt)  # Top-k documentos

# 4. Construir prompt con contexto
context = "\n\n".join([doc.page_content for doc in relevant_docs])
full_prompt = f"""Contexto:
{context}

Pregunta:
{prompt}

Respuesta:"""

# 5. Generar respuesta
response = llm.invoke(full_prompt)
```

### Búsqueda por Similitud Coseno

Pinecone calcula automáticamente:

```
similitud = cos(θ) = (A · B) / (||A|| × ||B||)
```

Donde:
- **A**: Embedding de la pregunta
- **B**: Embedding de cada chunk en la base de datos

Retorna los chunks con mayor similitud (más cercanos al 1.0).

---

## Guía de Uso

### Requisitos Previos

1. **Crear archivo `cv_mariela.txt`** con el contenido del CV
2. **Obtener API Key de Pinecone**:
   - Registrarse en [pinecone.io](https://www.pinecone.io)
   - Crear un proyecto
   - Copiar API Key y Environment

### Instalación

```bash
pip install streamlit pinecone-client langchain langchain-community \
    langchain-huggingface langchain-pinecone sentence-transformers \
    transformers torch
```

### Ejecución

```bash
streamlit run app.py
```

### Configuración Inicial

1. Ingresar **Pinecone API Key** en la barra lateral
2. Ingresar **Environment** (ej: `gcp-starter`)
3. Definir nombre del índice (ej: `my-rag-index`)
4. Seleccionar modelo LLM (FLAN-T5 Base recomendado)
5. Ajustar parámetros de generación si es necesario

### Uso del Chat

1. Esperar a que se carguen los modelos (primera vez ~2-5 min)
2. Hacer clic en "Probar Recuperación" para verificar funcionamiento
3. Escribir preguntas en el chat
4. El sistema recuperará contexto y generará respuestas

### Ejemplo de Preguntas

```
- ¿Cuál es la experiencia laboral de María?
- ¿Dónde estudió?
- ¿Qué habilidades tiene?
- Describe su formación académica
```

---

## Parámetros de Generación

### Temperature (0.01 - 1.0)
- **Baja (0.1)**: Respuestas más deterministas y conservadoras
- **Alta (0.9)**: Respuestas más creativas y variables

### Top P (0.1 - 1.0)
- **Nucleus sampling**: Considera tokens que sumen hasta este percentil
- **0.95**: Balance recomendado

### Repetition Penalty (1.0 - 2.0)
- Penaliza tokens repetidos
- **1.1**: Previene bucles sin limitar expresividad

### Max Length (64 - 512)
- Tokens máximos a generar
- **256**: Adecuado para respuestas concisas

---

## Comparación de Modelos (Probados vs Final)

### Modelos Probados Durante el Desarrollo

| Modelo | Tipo | Método | Estado | Resultado | Razón de Eliminación |
|--------|------|--------|--------|-----------|---------------------|
| **Mistral-7B-Instruct-v0.2** | Causal | API Remota | ❌ **DESCARTADO** | 💀 Inviable | Timeouts, rate limits, tamaño prohibitivo |
| **FLAN-T5 Base** | Seq2Seq | API Remota → Local | ⚠️ API falló → ✅ **Local OK** | ⭐⭐ Éxito | Migrado a local |
| **FLAN-T5 Small** | Seq2Seq | Local | ✅ **En versión final** | ⭐ Funciona bien | Rápido y ligero |
| **FLAN-T5 Base** | Seq2Seq | Local | ✅ **En versión final** | ⭐⭐ Recomendado | Balance óptimo |
| **FLAN-T5 Large** | Seq2Seq | Local | ✅ **En versión final** | ⭐⭐⭐ Mejor calidad | Alta calidad |
| **GPT-2** | Causal | Local | ❌ **ELIMINADO** | 💀 No funciona | Respuestas irrelevantes/inventadas |
| **GPT-2 Medium** | Causal | Local | ❌ **ELIMINADO** | 💀 No funciona | Respuestas irrelevantes/inventadas |
| **mT5 Base** | Seq2Seq | Local | ❌ **ELIMINADO** | 💀 No funciona | Respuestas vacías/genéricas |

### Código Final: Solo FLAN-T5

```python
# VERSIÓN FINAL - Solo estos modelos están disponibles
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

# GPT-2 y mT5 fueron ELIMINADOS completamente del código
```

### Características Comparativas

| Característica | FLAN-T5 ✅ | Mistral-7B ❌ | GPT-2 ❌ | mT5 ❌ |
|---------------|-----------|--------------|---------|--------|
| Sigue instrucciones | ✅ Excelente | ✅ Excelente (pero inaccesible) | ❌ No | ❌ Sin fine-tuning |
| Respuestas factuales | ✅ Sí | ✅ Sí (pero inaccesible) | ❌ Inventa información | ❌ Respuestas vacías |
| Viabilidad API gratuita | ⚠️ Limitada | ❌ Timeouts constantes | ⚠️ Muy lenta | ⚠️ Limitada |
| Viabilidad local | ✅ Excelente | ❌ Requiere 16+ GB RAM | ✅ Funciona pero mal | ✅ Funciona pero mal |
| Tamaño del modelo | ✅ 1-3 GB | ❌ ~15 GB | ✅ 500 MB - 1.5 GB | ✅ 1-2 GB |
| Soporte español | ✅ Bueno | ✅ Excelente | ⚠️ Limitado | ✅ Bueno (pero no funcional) |
| Arquitectura para Q&A | ✅ Seq2Seq ideal | ✅ Causal (buena con instrucciones) | ❌ Causal inadecuada | ⚠️ Seq2Seq sin entrenar |
| Calidad RAG | ✅ Alta | 🤷 No pudo probarse | ❌ Muy baja | ❌ Baja |
| En versión final | ✅ **SÍ** | ❌ **NO** | ❌ **NO** | ❌ **NO** |

### ¿Por qué Mistral-7B fue descartado?

**Mistral-7B-Instruct-v0.2** es objetivamente un modelo superior a FLAN-T5, pero resultó **completamente inviable** para este proyecto:

#### Problemas con API Remota
```python
# Intento fallido
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=token
)
# Resultado: Timeouts de 60+ segundos, rate limits agotados en minutos
```

#### Problemas con Ejecución Local
- **Tamaño**: ~15 GB (vs 1-3 GB de FLAN-T5)
- **RAM necesaria**: 16+ GB solo para el modelo
- **Sin GPU**: 3-5 minutos por respuesta (vs 5-15 segundos de FLAN-T5)
- **Objetivo del proyecto**: Sistema gratuito y accesible en hardware modesto

#### Decisión Final
Aunque Mistral-7B hubiera dado mejores respuestas, fue **descartado por ser incompatible con los objetivos del trabajo práctico** (sistema gratuito y accesible).

---

## Comparación de Modelos

---

## Arquitectura Técnica Detallada

### Clase CustomHuggingFaceEmbeddings

```python
class CustomHuggingFaceEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Convierte múltiples textos a vectores
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        # Convierte una pregunta a vector
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
```

**Propósito**: Adaptar Sentence-Transformers a la interfaz de LangChain.

### Sistema de Caché

- `@st.cache_resource`: Mantiene modelos en memoria entre ejecuciones
- Botón de limpieza para recargar con nuevos parámetros
- Evita descargas repetidas (modelos se guardan en `~/.cache/huggingface`)

### Manejo de Errores

El sistema incluye validación en múltiples capas:

1. **Configuración**: Verifica API keys antes de iniciar
2. **Conexión**: Maneja errores de red con Pinecone
3. **Carga de modelos**: Fallback a tokenizador lento si falla rápido
4. **Generación**: Try-catch en el pipeline de chat

---

## Limitaciones Conocidas

1. **Plan Gratuito de Pinecone**:
   - Máximo 5 índices
   - 100,000 vectores por índice
   - Latencia variable

2. **Modelos Locales**:
   - FLAN-T5 Base requiere ~3GB RAM
   - Primera carga lenta (~2-5 min)
   - Sin GPU, la generación es lenta (5-15 seg/respuesta)

3. **Calidad de Respuestas**:
   - Depende de la calidad del chunking
   - Chunks muy pequeños pierden contexto
   	- Chunks muy grandes reducen precisión de búsqueda

---

## Limitaciones y Hardware

### Requerimientos de Hardware
El sistema está diseñado para funcionar en entornos locales, pero tiene ciertos requisitos mínimos:

*   **CPU**: Procesador moderno (Intel i5/i7 o Apple Silicon M1/M2/M3).
*   **RAM**: Mínimo **8 GB** (16 GB Recomendado). El modelo `flan-t5-base` consume aprox 1-2 GB en memoria, más el overhead de Python y Streamlit.
*   **GPU**: No es estrictamente necesaria (funciona en CPU), pero mejora significativamente la velocidad de inferencia. En equipos Mac con Apple Silicon, se utiliza la aceleración MPS (Metal Performance Shaders).

### Limitaciones del Modelo
*   **Context Window**: FLAN-T5 tiene un límite de entrada de 512 tokens. Esto restringe la cantidad de contexto que se puede pasar al modelo, lo que a veces puede limitar respuestas que requieran analizar textos muy extensos.
*   **Alucinaciones**: Aunque el RAG reduce este problema, los modelos pequeños como `base` aún pueden inventar información si el contexto recuperado es insuficiente.

---

## Conclusiones

4. **Multilingüe**:
   - FLAN-T5 funciona mejor en inglés
   - Español tiene buena calidad pero no óptima

---

## Mejoras Futuras

### Corto Plazo
- [ ] Agregar historial de conversación persistente
- [ ] Mostrar fuentes (chunks recuperados) en la respuesta
- [ ] Implementar filtros por relevancia (score threshold)
- [ ] Agregar métricas de evaluación (precisión, recall)

### Mediano Plazo
- [ ] Soporte para múltiples documentos (varios CVs)
- [ ] Fine-tuning de FLAN-T5 en dataset de CVs
- [ ] Implementar re-ranking de resultados
- [ ] Sistema de feedback para mejorar respuestas

### Largo Plazo
- [ ] Migrar a modelos más grandes (FLAN-T5 XL)
- [ ] Implementar hybrid search (vectorial + keyword)
- [ ] Despliegue en cloud con GPU
- [ ] API REST para integración externa

---

## Conclusiones

### ✅ Logros

1. **Sistema RAG funcional** con stack completamente gratuito
2. **Identificación y eliminación de modelos inadecuados** (GPT-2, mT5)
3. **Documentación exhaustiva de problemas** encontrados durante el desarrollo
4. **Migración exitosa de API remota a modelos locales**
5. **Interfaz intuitiva** con Streamlit
6. **Manejo robusto de errores** en Pinecone
7. **Configuración flexible** con solo modelos viables (FLAN-T5)

### 📚 Aprendizajes Clave

1. **No todos los LLMs son apropiados para RAG**:
   - Seq2Seq > Causal para Q&A
   - Instruction-tuned > Pre-trained
   - **GPT-2 y mT5 fueron descartados** tras pruebas exhaustivas

2. **La arquitectura del modelo importa más que el tamaño**:
   - FLAN-T5 Base > GPT-2 Large para esta tarea
   - **Los modelos causales no funcionan bien sin fine-tuning específico**

3. **APIs gratuitas tienen limitaciones severas**:
   - Migración a modelos locales fue necesaria
   - Trade-off: RAM vs estabilidad (vale la pena)

4. **El chunking es crítico**:
   - Balance entre contexto y precisión
   - Overlap evita pérdida de información

5. **Pinecone simplifica vector search**:
   - Pero tiene limitaciones en plan gratuito
   - Alternativas: FAISS, Chroma (locales)

6. **Iteración y prueba son fundamentales**:
   - Probar múltiples modelos fue esencial
   - Eliminar opciones no viables mejora la experiencia del usuario

### 🎯 Recomendación Final

Para un sistema RAG de producción sobre documentos en español:

1. **Embeddings**: `multilingual-e5-large` (mejora sobre MiniLM)
2. **LLM**: FLAN-T5 Large o XL (si hay recursos)
3. **Vector Store**: Pinecone (escalable) o Qdrant (autohosted)
4. **Chunking**: 500-1000 caracteres con 10% overlap
5. **Infraestructura**: GPU para generación rápida

---

## Referencias

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)

---

**Autor**: Ezequiel Caamaño