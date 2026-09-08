import pandas as pd
from sklearn.model_selection import train_test_split


def load_clean_dataset(filepath, drop_columns, to_num, categorical_col, binary_col):
    df = pd.read_csv(filepath)
    df = df.drop(columns=drop_columns)
    df[to_num] = pd.to_numeric(df[to_num], errors="coerce").fillna(0)

    df = pd.get_dummies(df, columns=categorical_col)
    df = pd.get_dummies(df, columns=binary_col, drop_first=True)
    return df


def split_churn(df, target="Churn_Yes", test_size=0.2, random_state=48):
    y = df[target]
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
