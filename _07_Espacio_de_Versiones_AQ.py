class VersionSpace:
    def __init__(self):
        self.hipotesis = []

    def add_hipotesis(self, hipotesis):
        self.hipotesis.append(hipotesis)

    def remove_inconsistent(self, ejemplo):
        self.hipotesis = [hip for hip in self.hipotesis if hip.consistente(ejemplo)]

    def predict(self, ejemplo):
        resultado = None
        for hip in self.hipotesis:
            if hip.consistente(ejemplo):
                if resultado is None:
                    resultado = hip.clasificacion(ejemplo)
                else:
                    # Si hay m�ltiples hip�tesis, no podemos predecir de manera �nica.
                    return None
        return resultado

class Hipotesis:
    def __init__(self, atributos, clasificacion):
        self.atributos = atributos
        self.clasificacion = clasificacion

    def consistente(self, ejemplo):
        return all(ejemplo[attr] == val for attr, val in self.atributos.items())

    def clasificacion(self, ejemplo):
        return self.clasificacion

# Ejemplo de uso
vs = VersionSpace()

# Agregar hip�tesis iniciales al espacio de versiones
vs.add_hipotesis(Hipotesis({'Color': 'Rojo', 'Forma': 'Redonda'}, 'Manzana'))
vs.add_hipotesis(Hipotesis({'Color': 'Verde', 'Forma': 'Alargada'}, 'Pera'))

# Ejemplos de entrenamiento
entrenamiento = [
    {'Color': 'Rojo', 'Forma': 'Redonda', 'Clasificacion': 'Manzana'},
    {'Color': 'Verde', 'Forma': 'Alargada', 'Clasificacion': 'Pera'}
]

# Entrenar el espacio de versiones con ejemplos de entrenamiento
for ejemplo in entrenamiento:
    vs.remove_inconsistent({'Color': ejemplo['Color'], 'Forma': ejemplo['Forma']})

# Predecir con el espacio de versiones
nuevo_ejemplo = {'Color': 'Rojo', 'Forma': 'Redonda'}
prediccion = vs.predict(nuevo_ejemplo)

if prediccion is not None:
    print(f"Predicci�n: {prediccion}")
else:
    print("No se puede predecir de manera �nica.")
