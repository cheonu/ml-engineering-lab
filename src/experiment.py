import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlflow
from evaluate import compute_metrics


def run_experiment(experiment_name, model, X_train, X_test, y_train, y_test, params, pos_label=1,run_name=None):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Train the model
        model.fit(X_train, y_train)
        # Generate true predictions on test data
        y_pred = model.predict(X_test)
        # Log metrics to MLflow
        metrics = compute_metrics(y_test, y_pred, pos_label)
       
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.sklearn.log_model(model, "model")

    print("Run successfully logged MLflow. backend!")
