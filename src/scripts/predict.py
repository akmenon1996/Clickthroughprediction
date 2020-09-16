import pandas as pd
import sys
import joblib
import os
from termcolor import colored
from feature_engineering import feature_engineer



def main():
    """
    Pipeline to predict Click Through Probability of a given customer.
    """
    filename = sys.argv[1]
    model = sys.argv[2]
    data = pd.read_csv(filename)
    predict(data,model)

def predict(data,model):
    """

    :param data: Data to be predicted in the pandas format.
    :param model: Filepath to the trained model.
    :return: None
    This function used the loaded model to make predictions
    on the click probability of the data and store as a log file.
    """
    print(colored("########PREDICTING DATA########", 'blue'))
    feature_engineered_data = feature_engineer(data)
    xgboost_model = joblib.load(model)
    # Adding Click Probability as new column.
    data['Click_Probability'] = [i[1] for i in
                                 xgboost_model.predict_proba(
                                     feature_engineered_data.drop('Response',axis=1))]
    os.makedirs(os.path.dirname("logs/logs.txt"), exist_ok=True)
    data.to_csv('logs/results.csv', index=False)
    print(colored("Finished. \n"
                  "Find logs in the logs folder!", 'blue'))

if __name__ == "__main__":
    global usage
    USAGE = """
            Usage: python {prog_name} <filename> <model_filename>"

            """.format(prog_name=sys.argv[0])

    if len(sys.argv) != 3:
        print(colored(USAGE,'red'))
    else:
        main()