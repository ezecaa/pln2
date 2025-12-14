# Documentación: Sistema RAG Multi-Agente con Pinecone (TP3)

## Índice
1. [Introducción](#introducción)
2. [Arquitectura del Sistema Multi-Agente](#arquitectura-del-sistema-multi-agente)
3. [Configuración de Agentes](#configuración-de-agentes)
4. [Lógica de Ruteo](#lógica-de-ruteo)
5. [Cambios y Soluciones Técnicas](#cambios-y-soluciones-técnicas)
6. [Limitaciones y Hardware](#limitaciones-y-hardware)
7. [Guía de Uso](#guía-de-uso)

---

## Introducción

Este documento describe la extensión del sistema RAG original (TP2) hacia una **arquitectura multi-agente** (TP3). El objetivo es permitir consultas sobre múltiples perfiles (CVs) simultáneamente, detectando automáticamente a qué persona se refiere la pregunta o, en su defecto, utilizando un agente por defecto.

### Objetivos del Trabajo Práctico 3
- Implementar múltiples índices vectoriales (uno por CV/persona).
- Desarrollar un "Router" semántico o basado en keywords.
- Permitir consultas comparativas ("Juan vs Carlos").
- Mantener la arquitectura gratuita y local del TP2.

---

## Arquitectura del Sistema Multi-Agente

El sistema evoluciona de una estructura lineal a una estructura de enrutamiento:

```
Usuario → Pregunta
     ↓
[Router de Agentes] (Detección de Nombres/Alias)
     ↓
     ├─ Agente 1 (Mariela) ──> Índice Pinecone 1
     ├─ Agente 2 (Juan) ─────> Índice Pinecone 2
     ├─ Agente 3 (Carlos) ───> Índice Pinecone 3
     └─ Agente 4 (Alumno) ───> Índice Pinecone 4 (Default)
     ↓
[Agregador de Contexto] (Une chunks de los agentes seleccionados)
     ↓
[LLM (FLAN-T5)] (Genera respuesta comparativa o individual)
     ↓
Respuesta
```

### Componentes Nuevos

1. **Diccionario de Agentes**: Configuración centralizada de perfiles y alias.
2. **Router Lógico**: Función `get_relevant_context` que decide qué índices consultar.
3. **Multi-Index Loading**: Carga dinámica de múltiples `PineconeVectorStore`.

---

## Configuración de Agentes

El sistema define 4 agentes principales en `AGENTS_CONFIG`:

| Agente | Nombre Real | Archivo Fuente | Alias de Detección | Índice Pinecone |
|--------|-------------|----------------|-------------------|-----------------|
| **Alumno** | Ezequiel Caamano | `alumno.txt` | alumno, ezequiel, yo, mi | `agent-alumno` |
| **Mariela** | Mariela Rodríguez | `mariela.txt` | mariela, rodríguez, gabriela | `agent-mariela` |
| **Juan** | Juan Pérez | `juan.txt` | juan, pérez | `agent-juan` |
| **Carlos** | Carlos Gómez | `carlos.txt` | carlos, gómez | `agent-carlos` |

**Nota**: El agente "Alumno" actúa como fallback (por defecto) si no se menciona a nadie más.

---

## Lógica de Ruteo

La lógica implementada en `get_relevant_context` funciona así:

1. **Normalización**: Convierte la pregunta a minúsculas.
2. **Detección**: Recorre todos los `aliases` de cada agente.
   ```python
   if alias in query_lower:
       targeted_agents.append(key)
   ```
3. **Decisión**:
   - Si `targeted_agents` tiene elementos, consulta SOLO esos índices.
   - Si está vacío, agrega `["alumno"]` por defecto.
4. **Recuperación**:
   - Itera por los agentes seleccionados.
   - Recupera los top-3 chunks de cada uno.
   - Formatea el contexto con separadores claros:
     ```
     --- Información sobre Mariela Rodríguez ---
     [Chunk 1...]
     
     --- Información sobre Juan Pérez ---
     [Chunk 1...]
     ```

---

## Cambios y Soluciones Técnicas

### 1. Error de CacheReplayClosureError
**Problema**: Al usar `@st.cache_resource` en la función de carga de agentes, Streamlit fallaba si se intentaba mostrar elementos de UI (`st.spinner`, `st.success`) dentro de la función cacheada en re-ejecuciones.
**Solución**: Se eliminó el decorador de caché de `load_and_index_agent`. Como la verificación de existencia del índice en Pinecone es rápida, el impacto en rendimiento es despreciable y mejora la estabilidad.

### 2. Segmentation Fault (Crash)
**Problema**: Al igual que en el TP2, la combinación de librerías de ML en Mac (Torch/TensorFlow) causaba un cierre inesperado.
**Solución**: Se replicó la configuración de variables de entorno crítica **antes de los imports**:
```python
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
```

### 3. Rutas de Archivos
**Ajuste**: Se movieron los archivos `.txt` a la raíz de la carpeta `TP3` y se actualizó la configuración para apuntar directamente a `archivo.txt` en lugar de `cvs/archivo.txt`.

---

---

## Limitaciones y Hardware

### Requerimientos de Hardware (Modelos Multi-Agente)
El TP3 utiliza por defecto el modelo `flan-t5-large` para mejorar la capacidad de razonamiento en comparaciones, lo que eleva los requisitos respecto al TP2:

*   **CPU**: Se recomienda encarecidamente Apple Silicon (M1/M2/M3) o procesadores Intel i7 recientes.
*   **RAM**: Mínimo **16 GB**. El modelo `flan-t5-large` ocupa aprox 3-4 GB en memoria, y cargar múltiples índices de Pinecone añade overhead. Con 8 GB es posible que el sistema sea inestable o lento.
*   **Velocidad**: La inferencia es más lenta que en TP2. En CPU pura puede tardar 10-30 segundos por respuesta compleja. En Mac con MPS, esto se reduce a 5-10 segundos.

### Limitaciones del Modelo y Arquitectura
*   **Context Window**: Al igual que la versión Base, `flan-t5-large` tiene un límite de 512 tokens. Esto es crítico en el TP3 porque debemos "meter" información de múltiples agentes en ese espacio reducido.
    *   *Solución*: Implementamos un **Retrieval Adaptativo** (`k=2` para comparaciones) para evitar saturar la ventana, aunque esto sacrifica detalle profundo en favor de amplitud.
*   **Latencia de Red**: Al consultar múltiples índices de Pinecone secuencialmente, la latencia de red se suma.
*   **Precisión Comparativa**: Aunque el modelo es mejor, comparar 3 o más personas simultáneamente suele degradar la calidad de la respuesta debido a la mezcla de información en el contexto.

---

## Guía de Uso

1. **Ejecución**:
   ```bash
   cd TP3
   streamlit run chatbot-agents.py
   ```
2. **Configuración**:
   - Ingresa tus credenciales de Pinecone (igual que en TP2).
   - El sistema creará automáticamente los 4 índices si tienes espacio (Recuerda el límite de 5 índices del plan gratuito).

3. **Ejemplos de Consultas**:

   - **Consulta Directa (Mariela)**:
     > "¿Qué experiencia tiene **Mariela**?"
     -> *Consulta solo el índice `agent-mariela`.*

   - **Consulta Comparativa (Juan vs Carlos)**:
     > "Compara las habilidades de **Juan** y **Carlos**."
     -> *Consulta índices `agent-juan` y `agent-carlos`, combina información y el LLM genera la comparación.*

   - **Consulta Implícita (Alumno)**:
     > "¿Qué estudiaste?"
     -> *Al no detectar nombres, consulta el índice `agent-alumno` (Ezequiel).*

---
**Autor**: Ezequiel Caamaño