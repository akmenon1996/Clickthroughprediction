import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def merge_data(num,cat,y):
    data_engineered_merged = pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    data_engineered_merged = pd.concat([data_engineered_merged.reset_index(drop=True),
                                         y.reset_index(drop=True)], axis=1)
    return data_engineered_merged

def categorical_encoding(x_data_all):
    categorical_df = x_data_all.select_dtypes(include='object')
    cat_df = categorical_df.drop(['Customer', 'Effective To Date'], axis=1)
    cat_df_significant = cat_df[
        ['Coverage', 'Education', 'EmploymentStatus', 'Location Code', 'Marital Status', 'Policy', 'Renew Offer Type',
         'Vehicle Class', 'Vehicle Size']]
    cat_columns_significant = cat_df_significant.columns
    lb = LabelEncoder()
    for col in cat_df_significant[cat_columns_significant]:
        cat_df_significant[col] = lb.fit_transform(cat_df_significant[col])
        cat_df_significant[col] = cat_df_significant[col].astype('category')

    cat_df_significant = pd.get_dummies(cat_df_significant, drop_first=True)
    return cat_df_significant


def numerical_standarization(x_data):
    x_data_numeric = x_data.select_dtypes(include=['int64', 'float'])
    columns = x_data_numeric.columns
    ss = StandardScaler()
    x_data_numeric_normalized = ss.fit_transform(x_data_numeric)
    x_data_numeric_normalized = pd.DataFrame(x_data_numeric_normalized, columns=columns)
    x_data_numeric_normalized_significant = x_data_numeric_normalized[['Income', 'Monthly Premium Auto',
                                                                       'Total Claim Amount']]
    return x_data_numeric_normalized_significant,ss


def feature_engineer(data):
    data.Response = data.Response.apply(lambda x: 0 if x == 'No' else 1)
    x_data_all = data.drop(['Response'], axis=1)
    y_data = data['Response']
    x_data_numeric_normalized_significant,ss = numerical_standarization(x_data_all)
    cat_df_significant = categorical_encoding(x_data_all)
    merged_data = merge_data(x_data_numeric_normalized_significant,cat_df_significant,y_data)
    return merged_data,ss

