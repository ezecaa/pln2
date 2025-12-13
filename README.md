# Procesamiento de Lenguaje Natural II (PLN2)

Repositorio de Trabajos Prácticos para la materia **Procesamiento de Lenguaje Natural II** del **Curso de Especialización en Inteligencia Artificial de la Facultad de Ingeniería de la UBA (FIUBA)**.

Este repositorio contiene la implementacion de sistemas avanzados de NLP utilizando tecnologías como **LangChain**, **Pinecone**, **HuggingFace Transformers** y **Streamlit**.

---

## 📁 Estructura del Repositorio

### 1. [TP1_TinyGPT_MoE](TP1_TinyGPT_MoE/)
* **Tema:** Grandes Modelos de Lenguaje (LLMs) y Mixture of Experts (MoE).
* **Descripción:** Implementación de una arquitectura GPT reducida (TinyGPT) y exploración de la técnica Mixture of Experts para mejorar la eficiencia del modelo.

### 2. [TP2_RAG_Chatbot](TP2/)
* **Tema:** Sistema RAG (Retrieval-Augmented Generation) Chatbot.
* **Descripción:** Chatbot conversacional capaz de responder preguntas sobre un documento específico (CV) utilizando una base de datos vectorial.
* **Tecnologías:** Pinecone, LangChain, FLAN-T5 (Local), Streamlit.
* **Funcionalidades Clave:**
    * Embeddings multilingües.
    * Persistencia de vectores en Pinecone.
    * Chatbot interactivo con memoria de contexto simple.
    * Ejecución 100% local con modelos optimizados.
* **Video de Funcionamiento:**
    > [LINK_VIDEO_TP2_PENDIENTE]

### 3. [TP3_Multi_Agent_RAG](TP3/)
* **Tema:** Sistema RAG Multi-Agente con Ruteo Inteligente.
* **Descripción:** Evolución del TP2 hacia una arquitectura de agentes múltiples donde el sistema decide a qué "experto" (índice vectorial) consultar basándose en la pregunta del usuario.
* **Tecnologías:** Pinecone (Múltiples índices), Sentence-Transformers, Lógica de Ruteo, FLAN-T5.
* **Funcionalidades Clave:**
    * **Router Semántico/Keyword:** Detecta sobre qué perfil (Mariela, Juan, Carlos, Alumno) se está preguntando.
    * **Consultas Multi-Índice:** Capacidad de consultar y comparar información de múltiples agentes en una sola respuesta.
    * **Arquitectura Escalable:** Fácil adición de nuevos agentes/perfiles.
* **Video de Funcionamiento:**
    > [LINK_VIDEO_TP3_PENDIENTE]

---

## ⚙️ Requisitos Generales

Para ejecutar los proyectos TP2 y TP3, se requiere instalar las siguientes dependencias principales:

```bash
pip install streamlit pinecone-client langchain langchain-community langchain-huggingface langchain-pinecone sentence-transformers transformers torch
```

> **Nota:** Se recomienda utilizar un entorno virtual (venv o conda) para evitar conflictos de dependencias.

## 🚀 Cómo Ejecutar

1.  **Clonar el repositorio.**
2.  **Entrar a la carpeta del TP deseado (TP2 o TP3).**
3.  **Configurar credenciales:** Tener a mano la API Key de Pinecone.
4.  **Ejecutar la app de Streamlit:**

    ```bash
    # Para TP2
    cd TP2
    streamlit run chatbot.py

    # Para TP3
    cd TP3
    streamlit run chatbot-agents.py
    ```

---
**Curso de Especialización en Inteligencia Artificial - FIUBA**