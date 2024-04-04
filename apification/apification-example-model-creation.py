import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


def model_creation():
    # Generar datos aleatorios para la variable independiente (tamaño) y la variable dependiente (precio)
    np.random.seed(0)
    tamaño = np.random.randint(low=500, high=5000, size=(100,))
    habitaciones = np.random.randint(low=1, high=6, size=(100,))
    precio = 100000 + 200 * tamaño + 50000 * habitaciones + np.random.normal(0, 10000, size=(100,))
    # Crear un modelo de regresión lineal y ajustarlo a los datos
    modelo = LinearRegression()
    X = np.column_stack((tamaño, habitaciones))
    modelo.fit(X, precio)
    # Guardar el modelo en un archivo pickle
    with open('modelo.pkl', 'wb') as archivo:
        pickle.dump(modelo, archivo)


if __name__ == "__main__":
    model_creation()
    print("Done!")