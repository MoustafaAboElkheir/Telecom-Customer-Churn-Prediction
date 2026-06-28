"""Preprocessing pipeline for telecom churn dataset."""
import pandas as pd, numpy as np, argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def preprocess(data_path):
    df = pd.read_csv(data_path)
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
    X, y = df_enc.drop('Churn', axis=1), df_enc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    rus = RandomUnderSampler(random_state=42)
    X_train_b, y_train_b = rus.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_b)
    X_test_s  = scaler.transform(X_test)
    print(f"Train: {X_train_s.shape} | Test: {X_test_s.shape}")
    return X_train_s, X_test_s, y_train_b, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/telecom_churn.csv')
    args = parser.parse_args()
    preprocess(args.data)
