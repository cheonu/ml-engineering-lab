import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlflow
from sklearn.ensemble import RandomForestClassifier
from data import load_split
from evaluate import compute_metrics


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("breast-cancer")



# 3. Initialize the Random Forest Classifier
max_depth_val = 3
n_estimators_val = 100


rf_model = RandomForestClassifier(
    n_estimators=n_estimators_val, 
    max_depth=max_depth_val, 
    random_state=48, 
    n_jobs=-1)

with mlflow.start_run(run_name="parent-run") as parent_run:

    X_train, X_test, y_train, y_test = load_split()

    # Train the model
    rf_model.fit(X_train, y_train)

    # Generate true predictions on test data
    y_pred = rf_model.predict(X_test)


    # Log metrics to MLflow
    metrics = compute_metrics(y_test, y_pred)

    mlflow.log_metric("accuracy", metrics["accuracy"]) 
    mlflow.log_metric("f1_score", metrics["f1"])
    mlflow.log_metric("tn", metrics["tn"])
    mlflow.log_metric("fp", metrics["fp"])
    mlflow.log_metric("fn", metrics["fn"])
    mlflow.log_metric("tp", metrics["tp"])

    mlflow.log_param("max_depth", max_depth_val)
    mlflow.log_param("n_estimators", n_estimators_val)

    mlflow.sklearn.log_model(rf_model, "model")

print("Run successfully logged MLflow. backend!")






