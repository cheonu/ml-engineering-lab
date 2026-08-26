from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def  load_split(test_size=0.2, random_state=48):
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test

