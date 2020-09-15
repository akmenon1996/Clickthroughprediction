import pandas as pd
import numpy as np
import sys
import joblib
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


def main():
    filename = sys.argv[1]
    model = sys.argv[2]
    data = pd.read_csv(filename)
    predict(data)

def predict(data,model):
    feature_engineered_data, ss = feature_engineer(data)
    xgboost_model = joblib.load(model)
    data['Click_Probability'] = [i[1] for i in xgboost_model.predict_proba(feature_engineered_data.drop('Response',axis=1))]
    os.makedirs(os.path.dirname("logs/logs.txt"), exist_ok=True)
    data.to_csv('logs/results.csv', index=False)

if __name__ == "__main__":
    global usage
    USAGE = """
            Usage: python {prog_name} <filename> <model_filename>"

            """.format(prog_name=sys.argv[0])

    if len(sys.argv) != 3:
        print(USAGE)
    else:
        main()