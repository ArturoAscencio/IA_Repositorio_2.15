import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar un conjunto de datos de ejemplo (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar la Mejor Hip�tesis Actual
best_hypothesis = None
best_accuracy = 0.0

# Realizar un bucle para entrenar diferentes modelos y seleccionar el mejor
for _ in range(10):  # En este ejemplo, realizaremos 10 iteraciones para obtener diferentes modelos
    # Crear y entrenar un modelo (clasificador de regresi�n log�stica)
    model = LogisticRegression(solver='liblinear', multi_class='auto')
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular la precisi�n del modelo actual
    accuracy = accuracy_score(y_test, y_pred)

    # Actualizar la Mejor Hip�tesis Actual si encontramos un modelo con mejor precisi�n
    if accuracy > best_accuracy:
        best_hypothesis = model
        best_accuracy = accuracy

# Imprimir la precisi�n de la Mejor Hip�tesis Actual
print(f"Mejor Precisi�n: {best_accuracy}")
