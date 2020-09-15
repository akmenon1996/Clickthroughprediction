import os
import joblib
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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


