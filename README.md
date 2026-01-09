# 🌸 Clasificación de Flores Iris con IA 

🛠️ Metodología y Código
El desarrollo se centra en el archivo preparacion_Detector.py, el cual sigue estos pasos fundamentales del Deep Learning:

1. Configuración y Compatibilidad
El código fuerza el uso de tf_keras para asegurar la compatibilidad entre las versiones más recientes de TensorFlow y las funciones de preprocesamiento de imágenes:

os.environ['TF_USE_LEGACY_KERAS'] = '1'

2. Aumento de Datos (Data Augmentation)
Para evitar el sobreajuste (overfitting), el modelo no solo ve las fotos originales, sino versiones modificadas (rotadas, con zoom y volteadas horizontalmente). Esto obliga a la red a aprender la forma de la flor y no solo a memorizar una posición específica.

3. Transfer Learning (MobileNetV2)
En lugar de entrenar una red desde cero, utilizamos MobileNetV2 pre-entrenada con millones de imágenes (ImageNet).

Base Congelada: Se mantienen los "filtros" que ya saben reconocer colores y texturas.

Nueva Cabeza: Se añade una capa final de 17 neuronas con activación Softmax para clasificar nuestras especies específicas de nuestra dataset.

4. Entrenamiento y Salida
El modelo se compila con el optimizador Adam y se entrena durante 10 épocas, guardando finalmente el "cerebro" resultante en un archivo de alta jerarquía: modelo_flores_cnn.h5.

## Este proyecto utiliza Redes Neuronales Convolucionales (CNN) y el método de Transfer Learning para clasificar 17 categorías diferentes de flores. El modelo ha sido entrenado utilizando la arquitectura MobileNetV2, optimizada para identificar patrones visuales complejos con alta eficiencia.

- *Análisis de Caso Real* 
Para poner a prueba el modelo, se realizó una predicción con una imagen externa al dataset original:
! [lirio de san antonio](lirios_de_san_antonio_lilium_candidum.jpg)

* -Resultado de la IA
Clasificación: Windflower  
Certeza: 42.20%

Interpretación del Resultado
El modelo identificó la flor como Windflower con una confianza baja del 42.20%.

¿Por qué este resultado? La imagen cargada corresponde a un Lilium candidum (Lirio), una especie que no forma parte de las 17 categorías del dataset Oxford original.

Conclusión: La IA demuestra un comportamiento correcto al no asignar una certeza alta (como un 90%), indicando que la imagen no encaja perfectamente en sus categorías conocidas, pero seleccionando la opción visualmente más similar disponible.