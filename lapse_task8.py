import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# 1. Load data and recreate the Task 6 split

df = pd.read_csv("outputs/lapse_features.csv")

target = "lapsed"
id_col = "customer_id"

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

categorical_features = [
    "favourite_category",
    "city",
    "gender",
]

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Keep test customer IDs for the retention list.
test_customer_ids = df.loc[X_test.index, id_col].reset_index(drop=True)


# 2. Build preprocessing pipelines

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# 3. Refit Logistic Regression GridSearchCV

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

logistic_search.fit(X_train, y_train)

best_logistic = logistic_search.best_estimator_

print("Best Logistic Regression parameters:")
print(logistic_search.best_params_)


# 4. Logistic Regression coefficient interpretation

feature_names = best_logistic.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = best_logistic.named_steps[
    "model"
].coef_[0]

coefficient_table = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
    }
)

coefficient_table["absolute_coefficient"] = (
    coefficient_table["coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    "absolute_coefficient",
    ascending=False,
)

print("\nTop Logistic Regression coefficients:")
print(
    coefficient_table[
        ["feature", "coefficient"]
    ].head(10).to_string(index=False)
)

print("\nThree coefficient signs to interpret:")
print(
    coefficient_table[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)


# 5. Refit Random Forest GridSearchCV

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nBest Random Forest parameters:")
print(rf_search.best_params_)


# 6. Top five Random Forest feature importances

rf_feature_names = best_rf.named_steps[
    "preprocessor"
].get_feature_names_out()

rf_importances = best_rf.named_steps[
    "model"
].feature_importances_

importance_table = pd.DataFrame(
    {
        "feature": rf_feature_names,
        "importance": rf_importances,
    }
).sort_values(
    "importance",
    ascending=False,
)

print("\nTop 5 Random Forest feature importances:")
print(
    importance_table.head(5).to_string(index=False)
)


# 7. Save selected Logistic Regression model

model_path = "outputs/lapse_model.joblib"

joblib.dump(best_logistic, model_path)

print(f"\nSaved selected model to: {model_path}")


# 8. Reload model and verify predictions

reloaded_model = joblib.load(model_path)

original_predictions = best_logistic.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions,
)

print("\nReload verification:")
print("Predictions match exactly:", predictions_match)

if not predictions_match:
    raise RuntimeError(
        "Reloaded model predictions do not match!"
    )


# 9. Premium segment retention list

segments = pd.read_csv(
    "outputs/customer_segments.csv"
)

premium_ids = set(
    segments.loc[
        segments["is_premium"] == True,
        "customer_id",
    ]
)

test_results = pd.DataFrame(
    {
        "customer_id": test_customer_ids,
        "actually_lapsed": y_test.reset_index(drop=True),
        "lapse_probability": reloaded_model.predict_proba(
            X_test
        )[:, 1],
    }
)

premium_test_results = test_results[
    test_results["customer_id"].isin(premium_ids)
].copy()

retention_list = premium_test_results.sort_values(
    "lapse_probability",
    ascending=False,
).head(10)

retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed",
    ]
]

retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False,
)

print("\nPremium customers in test set:", len(premium_test_results))
print("\nTop Premium retention candidates:")
print(retention_list.to_string(index=False))

print(
    "\nActually lapsed among top retention candidates:",
    int(retention_list["actually_lapsed"].sum()),
)

print("\nSaved retention list to outputs/retention_list.csv")