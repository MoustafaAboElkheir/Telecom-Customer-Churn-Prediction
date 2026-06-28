"""Evaluation utilities for churn prediction models."""
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print(f"AUC-ROC: {roc_auc_score(y_test, proba):.4f}")
    print(classification_report(y_test, preds, target_names=['No Churn','Churn']))
    return {"auc": roc_auc_score(y_test, proba)}
