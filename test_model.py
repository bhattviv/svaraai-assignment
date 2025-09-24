import joblib

# Load vectorizer and models
vectorizer = joblib.load("tfidf_vectorizer.pkl")
log_reg = joblib.load("baseline_logreg_model.pkl")
lgbm = joblib.load("baseline_lgbm_model.pkl")

# Load label mapping
label_mapping = joblib.load("label_mapping.pkl")
reverse_mapping = {v: k for k, v in label_mapping.items()}

# Sample texts
sample_texts = ["I love this!", "This is terrible.", "Not sure about this."]

# Transform text
X_sample = vectorizer.transform(sample_texts)

# Logistic Regression predictions
pred_log = log_reg.predict(X_sample)
pred_log_labels = [reverse_mapping[p] for p in pred_log]
print("Logistic Regression Predictions:", pred_log_labels)

# LightGBM predictions
pred_lgb = lgbm.predict(X_sample).argmax(axis=1)
pred_lgb_labels = [reverse_mapping[p] for p in pred_lgb]
print("LightGBM Predictions:", pred_lgb_labels)
