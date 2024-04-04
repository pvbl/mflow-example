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
    ## ??
    # realizar predicciones en formato json

    resultado = "???"
    print(resultado)
    return jsonify(resultado)


if __name__ == '__main__':
    app.run(debug=True)