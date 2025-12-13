#!/usr/bin/env python3
"""
Script de prueba para verificar que sentence-transformers funciona sin TensorFlow
"""
import os

# Deshabilitar TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("🔧 Variables de entorno configuradas para deshabilitar TensorFlow")

try:
    print("\n📦 Importando sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers importado correctamente")
    
    print("\n📥 Cargando modelo (puede tardar unos minutos la primera vez)...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Modelo cargado correctamente")
    
    print("\n🧪 Probando embeddings...")
    test_text = "Hola, este es un texto de prueba"
    embedding = model.encode([test_text])
    print(f"✅ Embedding generado correctamente")
    print(f"   Dimensión: {embedding.shape}")
    print(f"   Primeros 5 valores: {embedding[0][:5]}")
    
    print("\n🎉 ¡Todo funciona correctamente!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
