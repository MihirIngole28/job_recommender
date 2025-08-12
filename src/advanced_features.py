# src/advanced_features.py
import fairlearn.metrics as fl_metrics
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier 
import shap
from sklearn.ensemble import RandomForestRegressor  # Proxy for explainer
import pandas as pd
import numpy as np

def bias_audit(recs_df, sensitive_feature='job_level'):
    # Group by sensitive feature (e.g., 'Mid senior' vs 'Associate')
    groups = recs_df[sensitive_feature].unique()
    group_counts = recs_df[sensitive_feature].value_counts(normalize=True)
    print("Demographic Parity Difference:", fl_metrics.demographic_parity_difference(
        np.ones(len(recs_df)),  # Dummy y_true (all 1s)
        recs_df['recommended'],  # y_pred (positional)
        sensitive_features=recs_df[sensitive_feature],
        method='between_groups'
    ))
    return group_counts

def debias_model(model, features, labels, sensitive_features):
    mitigator = ExponentiatedGradient(model, DemographicParity())
    mitigator.fit(features, labels, sensitive_features=sensitive_features)
    return mitigator


def explain_recs(model, features):
    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    shap.summary_plot(shap_values, features, feature_names=['sim_score', 'skill_match', 'location_match'])
    return shap_values



# Example
if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_jobs_tech_aug.csv')
    # Assu
    # me 'recommended' column (1 if in top-10, 0 else) from recs
    df['recommended'] = np.random.randint(0, 2, len(df))  # Synthetic for demo
    audit = bias_audit(df)
    print(audit)
    # Debias example (use your ranking NN)
    features = np.random.rand(len(df), 5)  # E.g., embedding sim + features
    labels = df['recommended']
    sensitive = df['job_level']
    mitigator = debias_model(RandomForestClassifier(), features, labels, sensitive)  # Example model

    # Proxy model (use your ranking NN or RF for demo).
    proxy_model = RandomForestRegressor().fit(features, labels)
    shap_values = explain_recs(proxy_model, features)
    # For output: "Matches 80% on skills like PyTorch (SHAP +0.5)"