import pandas as pd

def clean_data(df):
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(columns=['Cabin']) #drop cabin
    df['Sex'] = df['Sex'].map({'male':0, 'female':1}) #encode sex
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) #encode embarked
    return df

def missing_summary(df):
    """
    Returns a sorted summary of missing values per column.
    """
    return df.isnull().sum().sort_values(ascending=False)

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Plotting confusion matrices for all of our models
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix", save_path=None):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, feature_names, title="Feature Importance", save_path=None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 5))
        plt.barh(df_importance["Feature"], df_importance["Importance"], color='teal')
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
    else:
        print(f"{title}: Feature importance not available for this model.")



