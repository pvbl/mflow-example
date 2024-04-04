import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import sys


def train_test_split_data():
    # Cargar el dataset de Iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # Para simplificar, vamos a convertirlo en un problema binario solo con dos clases
    # Eliminaremos la clase '2' del dataset
    is_binary_class = y != 2
    X_binary, y_binary = X[is_binary_class, :], y[is_binary_class]

    # Dividir el dataset en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 

def train_model(train_x,train_y ):
    lr = LogisticRegression(solver='liblinear',n_jobs=1 )
    lr.fit(train_x, train_y)
    return lr

def eval_metrics(y_test, y_pred):
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report



def mlflow_model( MODELS_PATH= "."): 
    mlflow.set_experiment("LR Experiment")

    mlflow.set_tracking_uri("file:{0}/mlruns".format(MODELS_PATH))   
    with mlflow.start_run():
        X_train, X_test, y_train, y_test= train_test_split_data()
        lr = train_model(X_train,y_train)
        predicted_qualities = lr.predict(X_test)
        (accuracy, conf_matrix, class_report) = eval_metrics(y_test, predicted_qualities)
        score = lr.score(X_test, y_test)
        params = lr.get_params(deep=True)



        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("score", score)

        mlflow.log_params(params)

        mlflow.sklearn.log_model(lr, "model")

def main():
    mlflow_model()


if __name__ == "__main__":
    main()