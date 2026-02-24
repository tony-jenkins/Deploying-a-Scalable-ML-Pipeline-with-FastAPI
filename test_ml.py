import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import train_model, compute_model_metrics


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _make_tiny_df():
    """Small deterministic dataset with all required categorical columns + label."""
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28],
            "fnlgt": [77516, 83311, 215646, 234721, 338409],
            "education-num": [13, 13, 9, 7, 13],
            "capital-gain": [2174, 0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40, 40],
            "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private", "Private"],
            "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
            ],
            "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband", "Wife"],
            "race": ["White", "White", "White", "Black", "Black"],
            "sex": ["Male", "Male", "Male", "Male", "Female"],
            "native-country": ["United-States", "United-States", "United-States", "United-States", "Cuba"],
            "salary": ["<=50K", ">50K", "<=50K", "<=50K", ">50K"],
        }
    )


def test_process_data_outputs():
    """
    Verify process_data returns numpy arrays with expected row counts and non-null encoders.
    """
    df = _make_tiny_df()
    X, y, encoder, lb = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_type():
    """
    Verify train_model returns a trained LogisticRegression model.
    """
    df = _make_tiny_df()
    X, y, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")


def test_compute_model_metrics_known_case():
    """
    Verify compute_model_metrics returns perfect scores for perfect predictions.
    """
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0