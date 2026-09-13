import pandas as pd
import pytest

from data import load_clean_dataset, split_churn


def make_raw_csv(tmp_path, n_per_class=20):
    """Write a small fake raw dataset to a temp CSV and return its path.

    Mirrors the real Telco schema for the columns the pipeline touches:
    - customerID (should be dropped)
    - TotalCharges as strings, including a blank " " (should become 0.0)
    - Churn as Yes/No (should map to 1/0)
    Builds an equal number of churn / non-churn rows so the stratified
    split has enough of both classes to preserve the ratio.
    """
    rows = []
    for i in range(n_per_class):
        # non-churner
        rows.append({"customerID": f"no{i}", "TotalCharges": f"{100 + i}.5", "Churn": "No"})
        # churner
        rows.append({"customerID": f"yes{i}", "TotalCharges": f"{200 + i}.5", "Churn": "Yes"})

    df = pd.DataFrame(rows)
    # inject the blank-string quirk the real data has (new customer, no bill yet)
    df.loc[0, "TotalCharges"] = " "

    csv_path = tmp_path / "fake.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def clean_df(tmp_path):
    csv_path = make_raw_csv(tmp_path)
    return load_clean_dataset(str(csv_path), ["customerID"], "TotalCharges")


def test_drops_id_column(clean_df):
    assert "customerID" not in clean_df.columns


def test_totalcharges_is_numeric(clean_df):
    assert clean_df["TotalCharges"].dtype == "float64"


def test_no_missing_totalcharges(clean_df):
    # the blank " " should have been coerced to NaN then filled with 0
    assert clean_df["TotalCharges"].isna().sum() == 0
    assert (clean_df["TotalCharges"] == 0).sum() == 1


def test_churn_mapped_to_binary(clean_df):
    assert set(clean_df["Churn"].unique()) == {0, 1}


def test_stratified_split_preserves_ratio(clean_df):
    X_train, X_test, y_train, y_test = split_churn(
        clean_df, target="Churn", test_size=0.2, random_state=48
    )
    assert abs(y_train.mean() - y_test.mean()) < 0.05
