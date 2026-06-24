"""
churn_analysis.py

Customer Churn Analysis - Internship Project
----------------------------------------------
Goal: Telecom customers ko unke behaviour ke basis pe segments mein baatna
(clustering) aur fir yeh predict karna ki kaunsa customer churn (company
chhod sakta hai) karega (classification).

Two parts:
1. Customer Segmentation using K-Means
2. Churn Prediction using Random Forest (with hyperparameter tuning)

Author: Pranshul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, classification_report
)

import os

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
sns.set_style("whitegrid")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------
print("Step 1: Loading dataset...")
df = pd.read_csv("data/customer_churn.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.info())


# ------------------------------------------------------------------
# 2. Customer Segmentation (K-Means Clustering)
# ------------------------------------------------------------------
print("\nStep 2: Customer Segmentation using K-Means")

segmentation_data = df[['Monthly Charges', 'Tenure Months', 'Total Charges', 'Churn Probability']].copy()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)

# Elbow method to find optimal number of clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.savefig(f"{OUTPUT_DIR}/elbow_method.png")
plt.close()
print("Elbow plot saved -> outputs/elbow_method.png")

# Based on the elbow curve, 3 clusters gives a good balance
final_k = 3
kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
segmentation_data['Cluster'] = clusters

# Understand each cluster by checking average values
cluster_summary = segmentation_data.groupby('Cluster').mean()
print("\nCluster Summary (averages):")
print(cluster_summary)

# Name clusters based on cluster_summary (this needs to be checked manually
# every time data changes, since cluster numbers can shuffle on rerun)
cluster_order = cluster_summary['Churn Probability'].sort_values().index.tolist()
cluster_names = {
    cluster_order[0]: 'Loyal Low-Risk Customers',
    cluster_order[1]: 'Moderate Risk Customers',
    cluster_order[2]: 'High Risk Customers'
}
segmentation_data['Cluster Segment'] = segmentation_data['Cluster'].map(cluster_names)

# Merge segment info back to main dataframe for later use
df['Cluster Segment'] = segmentation_data['Cluster Segment']

print("\nCustomer count per segment:")
print(df['Cluster Segment'].value_counts())

# Visualize segments
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Monthly Charges', y='Churn Probability', hue='Cluster Segment',
                 data=df, palette='Spectral')
plt.title("Monthly Charges vs Churn Probability by Segment")
plt.savefig(f"{OUTPUT_DIR}/segment_monthly_charges.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Tenure Months', y='Churn Probability', hue='Cluster Segment',
                 data=df, palette='Spectral')
plt.title("Tenure vs Churn Probability by Segment")
plt.savefig(f"{OUTPUT_DIR}/segment_tenure.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total Charges', y='Churn Probability', hue='Cluster Segment',
                 data=df, palette='Spectral')
plt.title("Total Charges vs Churn Probability by Segment")
plt.savefig(f"{OUTPUT_DIR}/segment_total_charges.png")
plt.close()
print("Segment scatter plots saved in outputs/ folder")


# ------------------------------------------------------------------
# 3. Churn Prediction (Random Forest Classification)
# ------------------------------------------------------------------
print("\nStep 3: Churn Prediction using Random Forest")

# Encode categorical columns
model_df = df.copy()
model_df = pd.get_dummies(model_df, columns=['Contract Type', 'Internet Service'], drop_first=True)

feature_cols = [c for c in model_df.columns if c not in
                 ['CustomerID', 'Churn', 'Cluster Segment', 'Churn Probability']]

X = model_df[feature_cols]
Y = model_df['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameter search - trying different combinations manually
n_estimators_list = [100, 200, 300]
max_depth_list = [5, 10, 15]
results = []

print("Running hyperparameter search (this may take a few seconds)...")
for n_trees in n_estimators_list:
    for depth in max_depth_list:
        rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, Y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)

        results.append({
            'Trees': n_trees,
            'Depth': depth,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1
        })

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by=['Recall', 'Accuracy'], ascending=False)
print("\nTop hyperparameter combinations (sorted by Recall, since missing a")
print("churn-prone customer is costlier than a false alarm):")
print(result_df.head(10))
result_df.to_csv(f"{OUTPUT_DIR}/hyperparameter_results.csv", index=False)

# Pick the best combination automatically
best_row = result_df.iloc[0]
best_trees = int(best_row['Trees'])
best_depth = int(best_row['Depth'])
print(f"\nBest combination found -> Trees: {best_trees}, Depth: {best_depth}")

# Train final model with best parameters
final_model = RandomForestClassifier(
    n_estimators=best_trees,
    max_depth=best_depth,
    random_state=42,
    class_weight='balanced'
)
final_model.fit(X_train, Y_train)
y_pred_final = final_model.predict(X_test)

print("\nFinal Model - Classification Report:")
print(classification_report(Y_test, y_pred_final))

# Feature importance - useful insight for the business
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop features driving churn:")
print(importance_df.head(10))

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Top 10 Features Driving Churn")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()
print("Feature importance plot saved -> outputs/feature_importance.png")

print("\nDone. Saare results 'outputs/' folder mein save ho gaye hain.")
