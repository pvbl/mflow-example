import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pickle
import pandas as pd
from flask import Flask, jsonify, request
import pickle





from flask import Flask, jsonify, request

import pickle

# Cargar el modelo desde un archivo pickle
with open('modelo.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predecir_precio():
    # Recibir los datos de entrada desde el usuario
    datos = request.get_json()
    print(datos)

    # Procesar los datos y hacer la predicción del modelo
    tamaño = datos['tamano']
    habitaciones = datos['habitaciones']
    precio_predicho = modelo.predict([[tamaño, habitaciones]])

    # Devolver la predicción al usuario
    resultado = {'precio_predicho': precio_predicho[0]}
    return jsonify(resultado)


if __name__ == '__main__':
    app.run(debug=True)