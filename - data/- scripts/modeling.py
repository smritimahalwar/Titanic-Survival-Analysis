import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from scripts.utils import clean_data
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load and clean data
df = pd.read_csv("../data/train.csv")
df_clean = clean_data(df)

# Define target
y = df_clean['Survived']

# Select numeric features only
X = df_clean.select_dtypes(include=['int64', 'float64']).drop(columns=['PassengerId','Survived'], errors='ignore') #preventing leakage

# Impute missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
# Initialize to avoid PyCharm warnings
X_train = X_test = y_train = y_test = None

for train_index, test_index in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

from scripts.utils import evaluate_model, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

results = []

print("\n🔍 Model Evaluation Results:")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate and print
    evaluate_model(name, model, X_test, y_test)

    # Confusion matrix (show + save)
    plot_confusion_matrix(model, X_test, y_test,
                          title=f"{name} Confusion Matrix",
                          save_path=f"outputs/figures/{name}_confusion.png")


    # Append results for summary table
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Class 1)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Class 1)": recall_score(y_test, y_pred, pos_label=1),
        "F1-Score (Class 1)": f1_score(y_test, y_pred, pos_label=1)
    })

# Create summary DataFrame
df_results = pd.DataFrame(results)
print("\n📊 Model Comparison Summary:")
print(df_results.round(4))
# Export summary table to Excel
print("📁 Exporting to Excel…")
df_results.to_excel("model_comparison_summary.xlsx", index=False)
print("✅ Excel file exported.")

### 📊 Titanic Model Comparison Summary
'''
| Model               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|---------------------|----------|---------------------|------------------|--------------------|
| Logistic Regression | 1.0000   | 1.0000              | 1.0000           | 1.0000             |
| Random Forest       | 1.0000   | 1.0000              | 1.0000           | 1.0000             |
| SVM                 | 1.0000   | 1.0000              | 1.0000           | 1.0000             |
| KNN                 | 0.9888   | 0.9853              | 0.9710           | 0.9853             |
| XGBoost             | 1.0000   | 1.0000              | 1.0000           | 1.0000             |
'''

# Get feature names from X (before scaling)
from scripts.utils import plot_feature_importance
feature_names = X.columns

# Feature importance plots
# Plot for Random Forest
plot_feature_importance(models["Random Forest"], feature_names,
                        title="Random Forest Feature Importance",
                        save_path="outputs/figures/random_forest_importance.png")

# Plot for XGBoost
plot_feature_importance(models["XGBoost"], feature_names,
                        title="XGBoost Feature Importance",
                        save_path="outputs/figures/random_forest_importance.png")


# ... existing code that builds df_results ...

print("\n📊 Model Comparison Summary:")
print(df_results.round(4))

# --- Multi-sheet Excel export ---
import pandas as pd

with pd.ExcelWriter("model_comparison_summary.xlsx", engine="openpyxl") as writer:
    # Sheet 1: Model comparison summary
    df_results.to_excel(writer, sheet_name="Model Summary", index=False)

    # Sheet 2: Random Forest feature importance
    if hasattr(models["Random Forest"], "feature_importances_"):
        rf_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": models["Random Forest"].feature_importances_
        }).sort_values(by="Importance", ascending=False)
        rf_importance.to_excel(writer, sheet_name="RandomForest Importance", index=False)

    # Sheet 3: XGBoost feature importance
    if hasattr(models["XGBoost"], "feature_importances_"):
        xgb_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": models["XGBoost"].feature_importances_
        }).sort_values(by="Importance", ascending=False)
        xgb_importance.to_excel(writer, sheet_name="XGBoost Importance", index=False)

# CSV (quick peek in PyCharm/text editors)
df_results.to_csv("model_comparison_summary.csv", index=False)


print("✅ Multi-sheet Excel + CSV file exported.")

#FINAL PREDICTION EXPORT
import os
os.makedirs("outputs", exist_ok=True)
df_test = pd.read_csv("../test.csv")
df_test_clean = clean_data(df_test)

X_test_final = df_test_clean.select_dtypes(include=['int64','float64']).drop(columns=['PassengerId','Survived'], errors='ignore')
X_test_final = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)
X_test_final_scaled = scaler.transform(X_test_final)

best_model = models["Random Forest"]  # or whichever scored best
final_preds = best_model.predict(X_test_final_scaled)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": final_preds
})

submission.to_excel("titanic_predictions.xlsx", index=False)
submission.to_csv("titanic_predictions.csv", index=False)

print("✅ Final predictions exported to outputs/")
print("\n🔮 Final Predictions Preview:")
print(submission.head(10))


