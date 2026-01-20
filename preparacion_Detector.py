import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import tensorflow as tf
import numpy as np


try:
    from tf_keras.preprocessing.image import ImageDataGenerator
    from tf_keras.applications import MobileNetV2
    from tf_keras import layers, models
    print("✅ Cargado vía tf_keras")
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications import MobileNetV2
    from keras import layers, models
    print("✅ Cargado vía tensorflow.keras")
#  Configuración de rutas y parámetros
PATH_DATASET = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 

# Preprocesamiento y Data Augmentation

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

print("Cargando imágenes...")
train_gen = datagen.flow_from_directory(
    PATH_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    PATH_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Usamos MobileNetV2 de Google como base
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Congelamos el conocimiento previo

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), 
    layers.Dense(17, activation='softmax') # 17 neuronas = 17 tipos de flores
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Entrenamiento
print("\n🚀 Iniciando entrenamiento de la Red Neuronal...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

#  Guardar el modelo final
model.save('modelo_flores_cnn.h5')
print("\n✅ ¡Entrenamiento completado! Modelo guardado como 'modelo_flores_cnn.h5'")