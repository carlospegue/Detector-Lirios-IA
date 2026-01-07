# 🌸 Clasificación de Flores Iris con IA (SVM)

Este repositorio contiene un modelo de Inteligencia Artificial para la clasificación automática de especies de flores Iris utilizando **Machine Learning**. El modelo analiza dimensiones físicas (sépalo y pétalo) para distinguir entre las especies *Setosa*, *Versicolor* y *Virginica*.

## 🚀 Características del Proyecto
- **Algoritmo:** Support Vector Machines (SVM).
- **Optimización:** Búsqueda de hiperparámetros mediante `GridSearchCV`.
- **Precisión Lograda:** **98%** (Accuracy).
- **Persistencia:** El modelo se exporta en formato `.pkl` para su uso posterior sin necesidad de re-entrenamiento.

## 📊 Resultados del Modelo
El modelo fue evaluado con un 30% de los datos totales, obteniendo los siguientes resultados destacados:

- **Matriz de Confusión:** Solo una muestra de la especie *Versicolor* fue confundida, logrando una clasificación casi perfecta.
- **Métricas:** - Precisión promedio: 0.98
  - F1-Score: 0.98



## 🛠️ Requisitos
Para ejecutar este proyecto, necesitas tener instalado Python y las siguientes librerías:
```bash
pip install pandas seaborn scikit-learn matplotlib joblib