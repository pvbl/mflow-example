import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pickle
import pandas as pd
from flask import Flask, jsonify, request
import pickle




# Cargar el modelo desde un archivo pickle
with open('model.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)

app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # Recibir los datos de entrada desde el usuario
    datos = request.get_json()

    # Procesar los datos y hacer la predicción del modelo
    X1 = float(datos['X1'])
    X2 = float(datos['X2'])
    X3 = float(datos['X3'])
    X4 = float(datos['X4'])
    X5 = float(datos['X5'])

    out = modelo.predict([[X1,X2,X3,X4,X5]])

    # Devolver la predicción al usuario
    resultado = {'churn': str(out[0])}
    print(resultado)
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)