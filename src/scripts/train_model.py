import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os 
import seaborn as sns 
import sklearn as skl
from imblearn.over_sampling import SMOTE
from scipy.stats import norm 
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from numpy import loadtxt
from xgboost import XGBClassifier
from feature_engineering import feature_engineer
from data_engineering import data_engineer
from modelling import modelling
from predict import predict

def read_file(file):
    data = pd.read_csv(file)
    return data


def main():
    filename = sys.argv[1]
    data = read_file(filename)
    feature_engineered_data,ss = feature_engineer(data)
    x_train_res, y_train_res, x_test, y_test = data_engineer(feature_engineered_data)
    model_file = modelling(x_train_res, y_train_res, x_test, y_test)
    sample_data = data.sample(3000)
    predict(sample_data,model_file)


if __name__ == "__main__":
    global usage
    USAGE = """
        Usage: python {prog_name} <filename>"

        """.format(prog_name=sys.argv[0])

    if len(sys.argv) != 2:
        print(USAGE)
    else:
        main()