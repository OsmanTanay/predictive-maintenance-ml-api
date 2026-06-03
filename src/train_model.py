import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import joblib
from tensorflow import keras

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

    # Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Neural Network
    nn_model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
    print("\n=== Neural Network ===")
    print(classification_report(y_test, y_pred_nn))


if __name__ == "__main__":
    main()