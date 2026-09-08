from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# evaluate.py
def compute_metrics(y_test, y_pred,pos_label=1):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label)
    return {"accuracy": accuracy, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}




    

