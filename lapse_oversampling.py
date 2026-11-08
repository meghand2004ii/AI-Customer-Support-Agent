import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.utils import resample

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the lapse feature dataset
dataset = pd.read_csv("outputs/lapse_features.csv")

y = dataset["lapsed"]
X = dataset.drop(columns=["customer_id", "lapsed"])

# 2. Define features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
    "age"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# 3. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Combine training features and labels
train_data = X_train.copy()
train_data["lapsed"] = y_train

# Separate majority and minority classes
not_lapsed = train_data[
    train_data["lapsed"] == 0
]

lapsed = train_data[
    train_data["lapsed"] == 1
]

print("\nOriginal training class counts:")
print(train_data["lapsed"].value_counts().sort_index())

# 5. Oversample lapsed customers
# Keep original rows and add duplicates
lapsed_oversampled = resample(
    lapsed,
    replace=True,
    n_samples=len(not_lapsed),
    random_state=42
)

# Combine original majority and oversampled minority
balanced_train = pd.concat(
    [not_lapsed, lapsed_oversampled],
    ignore_index=True
)

print("\nBalanced training class counts:")
print(
    balanced_train["lapsed"]
    .value_counts()
    .sort_index()
)

# 6. Separate balanced X and y
X_train_balanced = balanced_train.drop(
    columns=["lapsed"]
)

y_train_balanced = balanced_train["lapsed"]

# 7. Preprocessing pipeline
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

# 8. Fit Logistic Regression on balanced training data
# Best C from the previous GridSearchCV
balanced_logistic = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=0.1,
                max_iter=1000
            )
        )
    ]
)

balanced_logistic.fit(
    X_train_balanced,
    y_train_balanced
)

# 9. Evaluate on the original, unchanged test set
balanced_predictions = balanced_logistic.predict(X_test)

print("\nBalanced Logistic Regression — TEST RESULTS")
print("------------------------------------------")

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            balanced_predictions,
            pos_label=1,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            balanced_predictions,
            pos_label=1,
            zero_division=0
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            balanced_predictions,
            pos_label=1,
            zero_division=0
        ),
        4
    )
)
