from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import sys


def download_raw_data(save=False):
    # Read the wine-quality csv file from the URL
    csv_url =\
    'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        raise ValueError(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    return data




def train_model(train_x,train_y,alpha,l1_ratio,random_state=42):
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)
    return lr
    


def traintest_split(data):
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return train_x,test_x,train_y,test_y



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2




def mlflow_model(data,alpha,l1_ratio, MODELS_PATH= "notebooks"): 
    mlflow.set_tracking_uri("file:{0}/mlruns".format(MODELS_PATH))   
    with mlflow.start_run():
        train_x,test_x,train_y,test_y = traintest_split(data)
        lr = train_model(train_x,train_y,alpha,l1_ratio)
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    data = download_raw_data()
    mlflow_model(data,alpha,l1_ratio)


if __name__ == "__main__":
    main()