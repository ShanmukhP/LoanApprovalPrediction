import pandas as pd
import joblib


def create_df():
    cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'TotalIncome']
    return pd.DataFrame(columns=cols, index=[0])


def fix_missing(df, na_dict):
    df.replace(na_dict, inplace=True)


def load_scaler():
    return joblib.load('./scaler/min-max.pkl')


def load_model():
    return joblib.load('./vot_hard.pkl')
