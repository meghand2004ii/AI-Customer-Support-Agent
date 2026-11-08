import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load the lapse feature dataset
dataset = pd.read_csv("outputs/lapse_features.csv")

# Target and identifiers
y = dataset["lapsed"]
X = dataset.drop(columns=["customer_id", "lapsed"])

# 2. Define numeric and categorical features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# 3. Confirm no feature column has missing values
missing_counts = X.isnull().sum()

print("\nMissing values per feature:")
print(missing_counts.to_string())

if X.isnull().any().any():
    raise ValueError(
        "Missing values detected in features. "
        "Resolve them before preprocessing."
    )

print("\nConfirmed: no feature column contains missing values.")

# 4. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain rows:", len(X_train))
print("Test rows:", len(X_test))

# 5. Create preprocessing pipelines
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

# 6. Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the fitted training preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing complete.")
print("Processed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

# 7. Compare frequency means

# Mean learned by the StandardScaler from training rows
frequency_scaler = (
    preprocessor
    .named_transformers_["numeric"]
    .named_steps["scaler"]
)

frequency_index = numeric_features.index("frequency")

training_frequency_mean = frequency_scaler.mean_[frequency_index]

# Mean frequency across the complete cohort
full_cohort_frequency_mean = X["frequency"].mean()

print("\nFrequency mean comparison:")
print(
    "Training-only scaler mean:",
    round(training_frequency_mean, 4)
)

print(
    "Full-cohort frequency mean:",
    round(full_cohort_frequency_mean, 4)
)

print(
    "\nThe means may differ because the scaler is fitted only "
    "on the training rows, while the full-cohort mean includes "
    "both training and test customers."
)