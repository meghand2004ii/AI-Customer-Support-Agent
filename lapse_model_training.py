
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the lapse feature dataset
dataset = pd.read_csv("outputs/lapse_features.csv")

y = dataset["lapsed"]
X = dataset.drop(columns=["customer_id", "lapsed"])

# 2. Feature groups
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

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Preprocessing pipeline
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

# 5. Baseline: always predict not lapsed
baseline_predictions = [0] * len(y_test)

print("\nBASELINE: Always predict not lapsed")
print("----------------------------------")
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Lapsed precision:", round(
    precision_score(
        y_test,
        baseline_predictions,
        pos_label=1,
        zero_division=0
    ), 4
))
print("Lapsed recall:", round(
    recall_score(
        y_test,
        baseline_predictions,
        pos_label=1,
        zero_division=0
    ), 4
))
print("Lapsed F1:", round(
    f1_score(
        y_test,
        baseline_predictions,
        pos_label=1,
        zero_division=0
    ), 4
))
print("Confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))

# 6. Logistic Regression GridSearchCV
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ]
)

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_search.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION")
print("-------------------")
print("Best parameters:", logistic_search.best_params_)
print("Best CV F1:", round(logistic_search.best_score_, 4))

# 7. Random Forest GridSearchCV
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

rf_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("\nRANDOM FOREST")
print("-------------")
print("Best parameters:", rf_search.best_params_)
print("Best CV F1:", round(rf_search.best_score_, 4))

# 8. Evaluate tuned models on untouched test set
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n{name} — TEST RESULTS")
    print("-" * (len(name) + 15))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Lapsed precision:", round(
        precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ), 4
    ))

    print("Lapsed recall:", round(
        recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ), 4
    ))

    print("Lapsed F1:", round(
        f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ), 4
    ))

    return predictions


logistic_predictions = evaluate_model(
    "Logistic Regression",
    logistic_search.best_estimator_,
    X_test,
    y_test
)

rf_predictions = evaluate_model(
    "Random Forest",
    rf_search.best_estimator_,
    X_test,
    y_test
)
