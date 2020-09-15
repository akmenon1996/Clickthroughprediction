import pandas as pd
import sys
import joblib
import os
from termcolor import colored
from feature_engineering import feature_engineer



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
        print(colored(USAGE,'red'))
    else:
        main()