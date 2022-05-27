from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, pos_label=1)