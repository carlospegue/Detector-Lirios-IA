import numpy as np
import tensorflow as tf
import os

# Forzamos el uso del Keras de compatibilidad para cargar el modelo
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from tf_keras.models import load_model
from tf_keras.preprocessing import image

# 1. Cargar el modelo
model = load_model('modelo_flores_cnn.h5')

# 2. Lista de etiquetas (Oxford 17)
clases = [
    "Bluebell", "Buttercup", "ColtsFoot", "Cowslip", "Crocus",
    "Daffodil", "Daisy", "Dandelion", "Fritillary", "Iris",
    "LilyValley", "Pansy", "Snowdrop", "Sunflower", "TigerLily",
    "Tulip", "Windflower"
]

def predecir(ruta):
    img = image.load_img(ruta, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    idx = np.argmax(preds)
    print(f"\n🌸 Flor: {clases[idx]} | ✅ Certeza: {preds[0][idx]*100:.2f}%")

# Prueba con una foto
predecir('image_0641.jpg')