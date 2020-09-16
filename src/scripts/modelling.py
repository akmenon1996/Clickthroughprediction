import os
import joblib
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


def xgboost_classifier(x_train_res, y_train_res, x_test, y_test):
    """
    :param x_train_res: Train Data
    :param y_train_res: Train Result
    :param x_test: Test Data
    :param y_test: Test Result
    :return: model file name
    Creates XGBoost Classification model and prints Confidence Interval.
    """
    xgbc = XGBClassifier()
    xgbc.fit(x_train_res, y_train_res) # Fitting the XGBoost model.
    xgbc_pred = xgbc.predict(x_test)
    print(confusion_matrix(xgbc_pred, y_test))
    print('Accuracy score:', accuracy_score(xgbc_pred, y_test))
    print(classification_report(xgbc_pred, y_test))
    cross_val_score_xgbc = cross_validate(xgbc, x_train_res, y_train_res, cv=5)
    print('Cross validation test_score', cross_val_score_xgbc['test_score'].mean())
    os.makedirs(os.path.dirname("models/models.txt"),exist_ok=True)
    filename = "models/xgboost_classifier.model"
    interval = 1.96 * sqrt((cross_val_score_xgbc['test_score'].mean() *
                            (1 - cross_val_score_xgbc['test_score'].mean())
                            )/y_train_res.shape[0]) #Binomial Proportions Confidence Interval
    print(colored(f"The Confidence Interval of the XGBoost classification model's F1 score is "
                  f"{cross_val_score_xgbc['test_score'].mean()} +/- {interval}",'green'))

    joblib.dump(xgbc,filename)
    return filename


def modelling(x_train_res, y_train_res, x_test, y_test):
    model_file = xgboost_classifier(x_train_res,y_train_res, x_test, y_test)
    print(colored("Done!",'green'))
    return model_file


