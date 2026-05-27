import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = "data/predictive_maintenance.csv"
MODEL_PATH = "models/random_forest_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    df_clean = df.drop(columns=["UDI", "Product ID", "Failure Type"])
    df_clean = pd.get_dummies(df_clean, columns=["Type"], drop_first=True)

    X = df_clean.drop("Target", axis=1)
    y = df_clean["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()