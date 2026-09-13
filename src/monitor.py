import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_clean_dataset
from evidently import Report
from evidently.presets import DataDriftPreset

def main():

    df = load_clean_dataset("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", ["customerID"], "TotalCharges")

    reference = df.sample(frac=0.5, random_state=1)
    current = df.drop(reference.index)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)
    result.save_html("drift_no_drift.html")
    print("drift_no_drift.html")

    drifted = current.copy()
    drifted["Contract"] = "Month-to-month"
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] + 30
    report2 = Report([DataDriftPreset()])
    result2 = report2.run(current_data=drifted, reference_data=reference)
    result2.save_html("drift_detected.html")
    print("Saved drift_detected.html")

if __name__ == "__main__":
    main()
