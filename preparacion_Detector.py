import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import glob
import json

try:
    from tf_keras.preprocessing.image import ImageDataGenerator
    from tf_keras.applications import MobileNetV2
    from tf_keras import layers, models
    from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
    print("✅ Cargado vía tf_keras")
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications import MobileNetV2
    from keras import layers, models
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    print("✅ Cargado vía tensorflow.keras")

# Configuración de rutas y parámetros
PATH_DATASET = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  # Aumentado para mejor entrenamiento

# Lista completa de las 17 clases esperadas (Oxford 17 Flowers)
CLASES_ESPERADAS = [
    "Bluebell", "Buttercup", "ColtsFoot", "Cowslip", "Crocus",
    "Daffodil", "Daisy", "Dandelion", "Fritillary", "Iris",
    "LilyValley", "Pansy", "Snowdrop", "Sunflower", "TigerLily",
    "Tulip", "Windflower"
]

# 1. Función para buscar imágenes con múltiples extensiones
def buscar_imagenes(carpeta):
    """Busca imágenes con cualquier extensión común"""
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', 
                   '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', 
                   '*.GIF', '*.TIFF', '*.webp', '*.WEBP']
    imagenes = []
    for ext in extensiones:
        imagenes.extend(glob.glob(os.path.join(carpeta, ext)))
    return imagenes

# 2. Explorar y verificar el dataset completo
def explorar_dataset(dataset_path='dataset'):
    """Explora el dataset y verifica que todas las clases tengan imágenes"""
    
    print("🔍 EXPLORANDO DATASET...")
    print("=" * 50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: No existe la carpeta '{dataset_path}'")
        return []
    
    all_items = os.listdir(dataset_path)
    class_names = []
    clases_encontradas = {}
    
    for item in all_items:
        item_path = os.path.join(dataset_path, item)
        
        # Solo procesar carpetas (no archivos) y excluir 'split'
        if os.path.isdir(item_path) and item != 'split':
            # Buscar imágenes
            imagenes = buscar_imagenes(item_path)
            
            if len(imagenes) > 0:
                class_names.append(item)
                clases_encontradas[item] = len(imagenes)
                print(f"  ✅ '{item}': {len(imagenes):>3} imágenes")
            else:
                print(f"  ⚠️  '{item}': 0 imágenes (será ignorada)")
    
    print(f"\n📊 RESUMEN: {len(class_names)} clases con imágenes encontradas")
    
    # Verificar clases faltantes
    clases_faltantes = []
    for clase in CLASES_ESPERADAS:
        if clase not in class_names:
            clases_faltantes.append(clase)
    
    if clases_faltantes:
        print(f"\n⚠️  ADVERTENCIA: {len(clases_faltantes)} clases esperadas NO encontradas:")
        for clase in clases_faltantes:
            print(f"   - {clase}")
        print("\n   Posibles soluciones:")
        print("   1. Verifica que los nombres de carpetas coincidan exactamente")
        print("   2. Asegúrate de que las carpetas tengan imágenes")
        print("   3. Las imágenes deben tener extensiones .jpg, .png, .jpeg, etc.")
    
    # Mostrar estadísticas
    print(f"\n📈 ESTADÍSTICAS:")
    total_imagenes = sum(clases_encontradas.values())
    print(f"   Total de imágenes: {total_imagenes}")
    print(f"   Promedio por clase: {total_imagenes//len(class_names) if class_names else 0}")
    
    return class_names, clases_encontradas

# 3. Organizar datos en train/val/test
def organizar_dataset_en_carpetas(dataset_path='dataset', class_names=None):
    """Organiza las imágenes en carpetas train/val/test automáticamente"""
    
    print("\n📁 ORGANIZANDO DATASET EN TRAIN/VAL/TEST (70/15/15)...")
    
    # Si no se proporcionan class_names, explorar el dataset
    if class_names is None:
        class_names, _ = explorar_dataset(dataset_path)
    
    # Si ya existe la carpeta 'split', eliminarla para empezar de nuevo
    split_path = os.path.join(dataset_path, 'split')
    if os.path.exists(split_path):
        print("🗑️  Eliminando organización previa...")
        shutil.rmtree(split_path)
    
    # Crear directorios si no existen
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            split_class_path = os.path.join(dataset_path, 'split', split, class_name)
            os.makedirs(split_class_path, exist_ok=True)
    
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    
    # Para cada clase, dividir las imágenes
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        
        # Obtener todas las imágenes de la clase
        images = buscar_imagenes(class_path)
        
        if len(images) == 0:
            print(f"❌ Error: Clase '{class_name}' no tiene imágenes válidas")
            continue
            
        # Si hay muy pocas imágenes, manejar de manera especial
        if len(images) < 10:
            print(f"⚠️  Clase '{class_name}' tiene pocas imágenes ({len(images)})")
            
            # Con pocas imágenes: 80% train, 10% val, 10% test
            if len(images) >= 3:
                train_files, temp_files = train_test_split(
                    images, test_size=0.2, random_state=42, shuffle=True
                )
                val_files, test_files = train_test_split(
                    temp_files, test_size=0.5, random_state=42, shuffle=True
                )
            else:
                # Si hay menos de 3, todas van a train
                train_files = images
                val_files, test_files = [], []
        else:
            # Dividir normal: 70% train, 15% val, 15% test
            train_files, temp_files = train_test_split(
                images, test_size=0.3, random_state=42, shuffle=True
            )
            val_files, test_files = train_test_split(
                temp_files, test_size=0.5, random_state=42, shuffle=True
            )
        
        # Copiar archivos a sus respectivas carpetas
        for file in train_files:
            dest = os.path.join(dataset_path, 'split', 'train', class_name, os.path.basename(file))
            shutil.copy2(file, dest)
            total_stats['train'] += 1
        
        for file in val_files:
            dest = os.path.join(dataset_path, 'split', 'val', class_name, os.path.basename(file))
            shutil.copy2(file, dest)
            total_stats['val'] += 1
        
        for file in test_files:
            dest = os.path.join(dataset_path, 'split', 'test', class_name, os.path.basename(file))
            shutil.copy2(file, dest)
            total_stats['test'] += 1
        
        print(f"✅ '{class_name}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    print(f"\n📊 TOTALES: {total_stats['train']} train, {total_stats['val']} val, {total_stats['test']} test")
    print("🎯 Organización completada exitosamente!")
    
    return total_stats

# 4. MAIN: Ejecutar todo el proceso
def main():
    print("=" * 60)
    print("🌺 ENTRENAMIENTO DE CLASIFICADOR DE FLORES (17 CLASES)")
    print("=" * 60)
    
    # Paso 1: Explorar el dataset
    class_names, clases_encontradas = explorar_dataset(PATH_DATASET)
    
    if not class_names:
        print("❌ ERROR: No se encontraron clases con imágenes.")
        print("   Verifica que el dataset esté en la carpeta 'dataset/'")
        return
    
    # Paso 2: Organizar en train/val/test
    stats = organizar_dataset_en_carpetas(PATH_DATASET, class_names)
    
    # Paso 3: Configurar rutas después de la división
    TRAIN_PATH = os.path.join(PATH_DATASET, 'split', 'train')
    VAL_PATH = os.path.join(PATH_DATASET, 'split', 'val')
    TEST_PATH = os.path.join(PATH_DATASET, 'split', 'test')
    
    # Verificar que existen imágenes en cada conjunto
    print("\n📊 VERIFICANDO CONJUNTOS DE DATOS:")
    for split, path in [('Train', TRAIN_PATH), ('Val', VAL_PATH), ('Test', TEST_PATH)]:
        if os.path.exists(path):
            total = 0
            for clase in os.listdir(path):
                clase_path = os.path.join(path, clase)
                if os.path.isdir(clase_path):
                    total += len(buscar_imagenes(clase_path))
            print(f"  {split}: {total} imágenes")
        else:
            print(f"  ❌ {split}: No existe la carpeta")
    
    # Paso 4: Preprocesamiento y Data Augmentation
    print("\n🔄 CONFIGURANDO PREPROCESAMIENTO...")
    
    # Solo aumento de datos para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.3,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Solo reescalado para validación y prueba
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("\n📊 CARGANDO IMÁGENES...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = test_val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    test_gen = test_val_datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # ¡Importante para evaluación!
    )
    
    # Paso 5: Mostrar información de las clases
    num_classes = train_gen.num_classes
    print(f"\n🎯 INFORMACIÓN DEL MODELO:")
    print(f"   - Número de clases: {num_classes}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Tamaño de imagen: {IMG_SIZE}")
    
    # Guardar el mapeo de clases
    class_indices = train_gen.class_indices
    # Invertir para tener {0: "Bluebell", 1: "Buttercup", ...}
    class_mapping = {v: k for k, v in class_indices.items()}
    
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"📝 Mapeo de clases guardado en 'class_mapping.json'")
    
    # Paso 6: Construir el modelo MobileNetV2
    print("\n🧠 CONSTRUYENDO MODELO MobileNetV2...")
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Congelar las capas base inicialmente
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Número dinámico de clases
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Mostrar resumen
    model.summary()
    
    # Paso 7: Callbacks para mejorar el entrenamiento
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'mejor_modelo.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Paso 8: Entrenamiento
    print("\n🚀 INICIANDO ENTRENAMIENTO...")
    print("=" * 50)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Paso 9: Fine-tuning (opcional pero recomendado)
    print("\n🎛️  INICIANDO FINE-TUNING...")
    
    # Descongelar las últimas capas de MobileNetV2
    base_model.trainable = True
    
    # Congelar las primeras capas
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continuar entrenamiento por pocas épocas
    history_fine = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Paso 10: Evaluar con el conjunto de prueba
    print("\n📈 EVALUANDO CON CONJUNTO DE PRUEBA...")
    test_results = model.evaluate(test_gen, verbose=1)
    
    print(f"\n🏆 RESULTADOS FINALES:")
    print(f"   📉 Pérdida (Loss): {test_results[0]:.4f}")
    print(f"   🎯 Precisión (Accuracy): {test_results[1]:.4f}")
    
    if len(test_results) > 2:
        print(f"   📊 Precisión (Precision): {test_results[2]:.4f}")
        print(f"   📈 Sensibilidad (Recall): {test_results[3]:.4f}")
    
    # Paso 11: Guardar el modelo final
    model.save('modelo_flores_cnn.h5')
    print("\n💾 Modelo final guardado como 'modelo_flores_cnn.h5'")
    
    # Guardar el historial de entrenamiento
    with open('historial_entrenamiento.json', 'w') as f:
        # Combinar ambos historiales si se hizo fine-tuning
        if 'history_fine' in locals():
            full_history = {}
            for key in history.history.keys():
                full_history[key] = history.history[key] + history_fine.history[key]
            json.dump(full_history, f, indent=2)
        else:
            json.dump(history.history, f, indent=2)
    
    print("📊 Historial guardado como 'historial_entrenamiento.json'")
    
    # Paso 12: Resumen final
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\n📋 RESUMEN:")
    print(f"   - Clases entrenadas: {num_classes}")
    print(f"   - Épocas totales: {EPOCHS + 5}")
    print(f"   - Precisión en test: {test_results[1]:.2%}")
    print(f"   - Archivos generados:")
    print(f"     1. modelo_flores_cnn.h5 (modelo principal)")
    print(f"     2. mejor_modelo.h5 (mejor versión durante entrenamiento)")
    print(f"     3. class_mapping.json (mapeo de clases)")
    print(f"     4. historial_entrenamiento.json (gráficos)")
    
    return model

# Ejecutar el programa principal
if __name__ == "__main__":
    try:
        model = main()
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL ENTRENAMIENTO: {e}")
        print("\n💡 POSIBLES SOLUCIONES:")
        print("   1. Verifica que todas las carpetas de flores tengan imágenes")
        print("   2. Asegúrate de tener permisos de escritura")
        print("   3. Revisa que las imágenes sean válidas (no corruptas)")
        print("   4. Verifica que tienes espacio en disco suficiente")
