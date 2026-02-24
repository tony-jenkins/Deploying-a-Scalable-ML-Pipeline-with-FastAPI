# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project implements a binary classification model that predicts whether an individual's income exceeds $50K per year based on census demographic and employment features. The model used is Logistic Regression from the scikit-learn library.

Categorical variables are transformed using OneHotEncoder, and the target variable (salary) is processed using LabelBinarizer. The model is trained on the UCI Census Income dataset.

## Intended Use

This model is intended for educational purposes to demonstrate an end-to-end machine learning pipeline, including preprocessing, model training, evaluation, slice-based performance analysis, and deployment using FastAPI.

The model predicts income classification (<=50K or >50K) based on demographic and employment attributes.

This model should not be used in real-world financial, hiring, eligibility, or policy decision-making scenarios.

## Training Data

The training data comes from the publicly available UCI Census Income dataset.

Key characteristics of the dataset:
    Approximately 32,561 records
    Mix of categorical and numerical features
    Target variable: salary
    Categorical features include workclass, education, marital status, occupation, relationship, race, sex, and native country
    Data split: 80% training / 20% testing using stratified sampling

Categorical features were encoded using one-hot encoding prior to training.

## Evaluation Data

The evaluation data consists of the 20% holdout test set created from the original dataset using stratified sampling to preserve label distribution.

Performance evaluation was conducted on:
    The full test dataset
    Slices of the test dataset based on categorical feature values

## Metrics

The model was evaluated using the following metrics:
    Precision
    Recall
    F1-score (F-beta with beta = 1)

Overall model performance on the test dataset:
    Precision: 0.7423
    Recall: 0.5676
    F1-score: 0.6433

These results indicate that the model is reasonably precise but has lower recall, meaning it fails to identify some positive (>50K income) cases.

Slice-based performance was also evaluated across all categorical features. For example:

For education = Doctorate:
    Precision: 0.9273
    Recall: 0.8095
    F1-score: 0.8644

Performance varies across slices, indicating that the model performs better on certain demographic groups than others.

Full slice metrics are recorded in slice_output.txt.

## Ethical Considerations

The dataset contains sensitive attributes such as race, sex, and education level. Using such features in predictive models can potentially reinforce historical and societal biases.

Slice-based performance analysis shows variability across demographic groups. If deployed in real-world applications, such a model could lead to unequal outcomes across protected groups.

This model is intended solely for instructional use and not for real-world decision-making.

## Caveats and Recommendations

Logistic Regression is a relatively simple linear model and may not capture complex nonlinear relationships in the data.

The recall score is lower than precision, indicating that some high-income individuals are not correctly identified.

The dataset may contain historical biases that influence model predictions.

Future improvements could include:
    Hyperparameter tuning
    Feature scaling for continuous variables
    Trying alternative models such as Random Forest or Gradient Boosting
    Bias detection and mitigation techniques
    Cross-validation instead of a single train/test split