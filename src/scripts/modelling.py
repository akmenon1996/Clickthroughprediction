import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import seaborn as sns
import sklearn as skl
from imblearn.over_sampling import SMOTE
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


def xgboost_classifier(x_train_res, y_train_res, x_test, y_test):
    xgbc = XGBClassifier()
    xgbc.fit(x_train_res, y_train_res)
    xgbc_pred = xgbc.predict(x_test)
    print(confusion_matrix(xgbc_pred, y_test))
    print('Accuracy score:', accuracy_score(xgbc_pred, y_test))
    print(classification_report(xgbc_pred, y_test))
    cross_val_score_rfc = cross_validate(xgbc, x_train_res, y_train_res, cv=5)
    print('Cross validation test_score', cross_val_score_rfc['test_score'].mean())
    os.makedirs(os.path.dirname("models/models.txt"),exist_ok=True)
    filename = "models/xgboost_classifier.model"
    joblib.dump(xgbc,filename)
    return filename


def modelling(x_train_res, y_train_res, x_test, y_test):
    model_file = xgboost_classifier(x_train_res,y_train_res, x_test, y_test)
    print("Done!")
    return model_file


