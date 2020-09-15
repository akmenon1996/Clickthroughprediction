from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_validate
import warnings
warnings.filterwarnings('ignore')


def data_engineer(data):
    X = data.drop(['Response'], axis=1)
    y = data['Response']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())


    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

    return x_train_res, y_train_res,x_test,y_test