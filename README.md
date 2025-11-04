# pln2
Repositorio Trabajos Prácticos de la materia Procesamiento de Lenguaje Natural II del Curso de Especialización en Inteligencia Artificial - FIUBA


# 📚 Proyectos de Procesamiento de Lenguaje Natural (PLN)

Este repositorio contiene los trabajos prácticos desarrollados para la asignatura de Procesamiento de Lenguaje Natural. Cada carpeta corresponde a un trabajo específico e incluye el código, los datos y los modelos generados.

---

## 📁 Estructura del Repositorio

El repositorio se divide en las siguientes carpetas, una por cada trabajo práctico:

### 1. [TP1_Clasificacion_Texto](TP1_Clasificacion_Texto/README.md)
* **Tema:** Clasificación de opiniones (Sentiment Analysis).
* **Herramientas/Modelos:** NLTK, Scikit-learn, Vectorización TF-IDF.
* **Contenido:** Notebooks de análisis exploratorio, preprocesamiento, entrenamiento y evaluación de modelos de clasificación.

### 2. [TP2_Modelos_Secuencia](TP2_Modelos_Secuencia/README.md)
* **Tema:** Reconocimiento de Entidades Nombradas (NER).
* **Herramientas/Modelos:** Keras/TensorFlow, Modelos RNN/LSTM/GRU.
* **Contenido:** Implementación de modelos secuenciales para tareas de etiquetado.

### 3. [TP3_Generacion_Lenguaje](TP3_Generacion_Lenguaje/README.md)
* **Tema:** Traducción Automática/Generación de texto.
* **Herramientas/Modelos:** Transformers, Hugging Face.
* **Contenido:** Experimentación con modelos pre-entrenados y fine-tuning.

---

## ⚙️ Cómo Ejecutar los Proyectos

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/tu_usuario/nombre_del_repositorio.git](https://github.com/tu_usuario/nombre_del_repositorio.git)
    ```
2.  **Instalar Dependencias:**
    * Cada proyecto puede tener un archivo `requirements.txt` en su carpeta.
    * Ejecuta: `pip install -r TP1_Clasificacion_Texto/requirements.txt`
3.  **Ejecución:**
    * Navega a la carpeta del proyecto y abre el notebook principal.

---

## 📝 Notas Adicionales

* **Datasets Grandes:** Los datasets que superan los 100MB no se suben directamente al repositorio (para evitar problemas de tamaño). En su lugar, se proporciona un enlace de descarga dentro del `README` de cada carpeta.
* **Modelos Grandes:** Se recomienda usar Git LFS para manejar archivos de modelos grandes.