import pandas as pd
import sys
from termcolor import colored
from feature_engineering import feature_engineer
from data_engineering import data_engineer
from modelling import modelling
from predict import predict

def read_file(file):
    data = pd.read_csv(file)
    return data


def main():
    filename = sys.argv[1]
    print(colored("########READING DATA########",'blue'))
    data = read_file(filename)
    print(colored("########PERFORMING FEATURE ENGINEERING########",'blue'))
    feature_engineered_data,ss = feature_engineer(data)
    print(colored("########PERFORMING DATA ENGINEERING########",'blue'))
    x_train_res, y_train_res, x_test, y_test = data_engineer(feature_engineered_data)
    print(colored("########PERFORMING DATA MODELLING########",'blue'))
    model_file = modelling(x_train_res, y_train_res, x_test, y_test)
    print(colored("########PREDICTING SAMPLE DATA########",'blue'))
    sample_data = data.sample(3000)
    predict(sample_data,model_file)
    print(colored("Finished. Find logs in the logs folder!",'blue'))


if __name__ == "__main__":
    global usage
    USAGE = """
        Usage: python {prog_name} <filename>"

        """.format(prog_name=sys.argv[0])

    if len(sys.argv) != 2:
        print(colored(USAGE,'red'))
    else:
        main()