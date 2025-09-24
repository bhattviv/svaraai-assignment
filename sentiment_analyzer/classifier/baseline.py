import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import joblib
import numpy as np

# Step 1: Load dataset
data = pd.read_csv("reply_classification_dataset.csv")

# Change "reply" and "label" to your dataset’s column names
X = data["reply"]
y = data["label"]

# Convert labels to numeric (LightGBM requirement)
# If labels are strings like "positive", "negative", "neutral", map them:
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y = y.map(label_mapping)

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Text preprocessing with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------
# Logistic Regression Model
# -------------------------
print("\n=== Logistic Regression ===")
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_tfidf, y_train)

y_pred_log = log_reg.predict(X_test_tfidf)
print("Accuracy (LogReg):", accuracy_score(y_test, y_pred_log))
print("F1 Score (LogReg):", f1_score(y_test, y_pred_log, average="weighted"))

# Save Logistic Regression model
joblib.dump(log_reg, "baseline_logreg_model.pkl")

# -------------------------
# LightGBM Model
# -------------------------
print("\n=== LightGBM ===")
lgb_train = lgb.Dataset(X_train_tfidf, label=y_train)
lgb_test = lgb.Dataset(X_test_tfidf, label=y_test, reference=lgb_train)

params = {
    "objective": "multiclass",
    "num_class": len(label_mapping),
    "metric": "multi_logloss",
    "verbosity": -1,
    "learning_rate": 0.1,
    "num_leaves": 31,
}

lgbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_test],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)],
)

y_pred_lgb = lgbm.predict(X_test_tfidf)
y_pred_lgb = np.array(y_pred_lgb).argmax(axis=1)

print("Accuracy (LightGBM):", accuracy_score(y_test, y_pred_lgb))
print("F1 Score (LightGBM):", f1_score(y_test, y_pred_lgb, average="weighted"))

# Save LightGBM model
joblib.dump(lgbm, "baseline_lgbm_model.pkl")

# Save vectorizer (common for both models)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# Save label mapping
joblib.dump(label_mapping, "label_mapping.pkl")
print("Models, vectorizer, and label mapping saved successfully!")