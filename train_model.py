import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
    inference,
)

# Use current working directory as the project root so script is portable
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print("Loading data from:", data_path)

# Load the census.csv data
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
# (stratify on label to keep distribution)
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])

# DO NOT MODIFY
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

# Process training data (fits encoder & lb)
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

# Train the model on the training dataset
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
lb_path = os.path.join(project_path, "model", "lb.pkl")
save_model(lb, lb_path)

# Load the model back (sanity)
model = load_model(model_path)

# Use the inference function to run model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Remove existing slice_output.txt if present so we create a fresh one
slice_output_file = os.path.join(project_path, "slice_output.txt")
if os.path.exists(slice_output_file):
    os.remove(slice_output_file)

# Compute performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features and every unique value
for col in cat_features:
    for slicevalue in sorted(test[col].dropna().unique()):
        count = test[test[col] == slicevalue].shape[0]
        p_s, r_s, fb_s = performance_on_categorical_slice(
            test, col, slicevalue, cat_features, "salary", encoder, lb, model
        )
        with open(slice_output_file, "a", encoding="utf-8") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p_s:.4f} | Recall: {r_s:.4f} | F1: {fb_s:.4f}", file=f)
            print("", file=f)
