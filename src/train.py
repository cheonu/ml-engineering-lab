import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import RandomForestClassifier
from data import load_clean_dataset, split_churn
from experiment import run_experiment
import yaml
import argparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_config(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main(config_path):
    cfg = load_config(config_path)

    # build the preprocessor: encode categoricals, pass numerics through
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["data"]["categorical_columns"]),
        ("num", "passthrough", cfg["data"]["numeric_columns"]),
    ])

    # build the pipeline: preprocessor + model as one object
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(**cfg["model"]["params"])),
    ])

    # load + split data from cfg["data"]
    df = load_clean_dataset(
        cfg["data"]["filepath"],
        cfg["data"]["drop_columns"],
        cfg["data"]["to_numeric"],
    )
    X_train, X_test, y_train, y_test = split_churn(df, target=cfg["data"]["target"])

    run_experiment(
        cfg["experiment_name"],
        pipeline,
        X_train, X_test, y_train, y_test,
        cfg["model"]["params"],
        pos_label=cfg["pos_label"],
        run_name=cfg["run_name"],
        registered_model_name=cfg["registered_model_name"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required = True)
    args = parser.parse_args()
    main(args.config)









