pip install Orange3

import Orange

# Cargar un conjunto de datos de ejemplo
data = Orange.data.Table("housing")

# Crear un árbol de regresión M5
tree = Orange.regression.M5RegressionLearner()
regression_tree = tree(data)

# Realizar predicciones
nuevo_ejemplo = data[0]  # Utilizamos el primer ejemplo del conjunto de datos
predicted_value = regression_tree(nuevo_ejemplo)
print(f"Predicción: {predicted_value}")
