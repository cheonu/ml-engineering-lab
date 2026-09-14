import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_clean_dataset
from monitor import check_drift
from train import main as train_main, load_config
from mlflow import MlflowClient
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


def main():
    df = load_clean_dataset(
       "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", ["customerID"], "TotalCharges" 
    )

    cfg = load_config("configs/churn.yaml")
    model_name = cfg["registered_model_name"]

    reference = df.sample(frac=0.5, random_state=1)
    current = df.drop(reference.index)

    if not check_drift(reference, current):
        print ("No significant drift detected. No retrain needed.")
        return

    print ("Drift detected. Retraining...")
    train_main("configs/churn.yaml")
    print("Retraining complete. New model version registered.")

    versions = client.search_model_versions(f"name='{model_name}'")
    new_version = max(versions, key=lambda v: int(v.version))
    new_f1 = client.get_run(new_version.run_id).data.metrics["f1"]

    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
        champ_f1 = client.get_run(champ.run_id).data.metrics["f1"]
    except Exception:
        champ_f1 = -1

    if new_f1 > champ_f1:
        client.set_registered_model_alias(model_name, "champion", new_version.version)
        print(f"Promoted v{new_version.version} (f1={new_f1:.3f} > {champ_f1:.3f})")
    else:
        print(f"Kept champion (new f1={new_f1:.3f} <= {champ_f1:.3f})")

if __name__ == "__main__":
    main()