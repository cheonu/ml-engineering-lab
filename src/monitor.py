import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_clean_dataset
from evidently import Report
from evidently.presets import DataDriftPreset


def drift_share(reference, current):
    """Run a data-drift report and return the share of drifted columns (0.0-1.0)."""
    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)
    d = result.dict()

    for metric in d["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return metric["value"]["share"]
    return 0.0


def check_drift(reference, current, threshold=0.5):
    """Return True if the share of drifted columns exceeds the threshold."""
    return drift_share(reference, current) > threshold


def main():
    df = load_clean_dataset(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", ["customerID"], "TotalCharges"
    )

    reference = df.sample(frac=0.5, random_state=1)
    current = df.drop(reference.index)

    # baseline: two halves of the same data -> little/no drift
    print(f"no-drift share:  {drift_share(reference, current):.2f}  drifted={check_drift(reference, current)}")

    drifted = current.copy()
    # shift many features to mimic a broadly changed customer base
    drifted["Contract"] = "Month-to-month"
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] + 30
    drifted["TotalCharges"] = drifted["TotalCharges"] + 500
    drifted["tenure"] = drifted["tenure"] + 20
    drifted["InternetService"] = "Fiber optic"
    drifted["PaymentMethod"] = "Electronic check"
    drifted["OnlineSecurity"] = "No"
    drifted["TechSupport"] = "No"
    drifted["StreamingTV"] = "Yes"
    drifted["Contract"] = "Month-to-month"
    drifted["PaperlessBilling"] = "Yes"
    drifted["SeniorCitizen"] = 1
    print(f"drifted share:   {drift_share(reference, drifted):.2f}  drifted={check_drift(reference, drifted)}")


if __name__ == "__main__":
    main()
