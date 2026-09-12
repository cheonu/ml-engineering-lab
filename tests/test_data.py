from data import load_clean_dataset, split_churn
import pytest

@pytest.fixture
def clean_df():
    return load_clean_dataset(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ["customerID"], "TotalCharges"
    )

cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaymentMethod"]
bin_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

def test_drops_id_column(clean_df):
    assert "customerID" not in clean_df.columns


def test_totalcharges_is_numeric(clean_df):
    assert clean_df["TotalCharges"].dtype == "float64"

def test_no_missing_totalcharges(clean_df):
    assert clean_df["TotalCharges"].isna().sum() == 0

def test_stratified_split_preserves_ratio(clean_df):
    X_train, X_test, y_train, y_test = split_churn (clean_df, target="Churn", test_size=0.2, random_state=48)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
