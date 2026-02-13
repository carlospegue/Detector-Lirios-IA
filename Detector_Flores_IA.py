
import numpy as np
import tensorflow as tf
import os
from tf_keras.models import load_model
from tf_keras.preprocessing import image

# 1. Cargar modelo
modelo = load_model('modelo_flores_cnn.h5')

# 2. Nombres de las 17 flores (en el orden del entrenamiento)
flores = [
    "Bluebell", "Buttercup", "ColtsFoot", "Cowslip", "Crocus",
    "Daffodil", "Daisy", "Dandelion", "Fritillary", "Iris",
    "LilyValley", "Pansy", "Snowdrop", "Sunflower", "TigerLily",
    "Tulip", "Windflower"
]

# 3. Función de detección
def que_flor_es(ruta):
    """Devuelve el nombre de la flor en la imagen"""
    # Cargar imagen
    img = image.load_img(ruta, target_size=(224, 224))
    # Preprocesar
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predecir
    prediccion = modelo.predict(img_array, verbose=0)
    # Encontrar flor con mayor probabilidad
    indice = np.argmax(prediccion[0])
    confianza = prediccion[0][indice] * 100
    
    return flores[indice], confianza

# 4. Uso
if __name__ == "__main__":
    # Pedir imagen
    print("🌺 DETECTOR DE FLORES")
    print("-" * 30)
    
    ruta_imagen = input("Ruta de la foto: ").strip()
    
    # Quitar comillas si las tiene
    ruta_imagen = ruta_imagen.strip('"').strip("'")
    
    if not os.path.exists(ruta_imagen):
        print(f"\n❌ No se encuentra: {ruta_imagen}")
    else:
        print("\n🔍 Analizando...")
        flor, porcentaje = que_flor_es(ruta_imagen)
        print(f"\n✅ RESULTADO: {flor} ({porcentaje:.1f}% seguro)")
    
    input("\nPresiona Enter para terminar...")
