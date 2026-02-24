# scripts/train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_model


def main():
    # Load data
    df = pd.read_csv("data/census.csv")

    # Categorical features used in the project
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Train / test split
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["salary"]
    )

    # Process training data (fits encoder & label binarizer)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # Process test data (use fitted encoder & lb)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate quickly
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Overall metrics -> precision: {precision:.4f}, "
        f"recall: {recall:.4f}, "
        f"fbeta: {fbeta:.4f}"
    )

    # Save artifacts to disk (model/, encoder/, lb/)
    save_model(model, "model/model.joblib")
    save_model(encoder, "model/encoder.joblib")
    save_model(lb, "model/lb.joblib")
    print("Saved model and encoders to model/")


if __name__ == "__main__":
    main()
