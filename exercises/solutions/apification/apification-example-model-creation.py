import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pickle
import pandas as pd
from flask import Flask, jsonify, request
import pickle


def model_creation():
    # Generar datos ficticios para el modelo
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                            n_redundant=0, n_classes=2, weights=[0.7, 0.3], random_state=42)
    X_train = pd.DataFrame(X,columns=["X1","X2","X3","X4","X5"])
    # Crear un modelo de regresión logística y ajustarlo a los datos
    modelo = LogisticRegression(random_state=42)
    modelo.fit(X_train, y)
    # Guardar el modelo en un archivo pickle
    with open('model.pkl', 'wb') as archivo:
        pickle.dump(modelo, archivo)


if __name__ == "__main__":
    model_creation()
    print("Done!")