import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# ✅ Load dataset
df = pd.read_csv("data/synthetic_microlending.csv")
print("✅ Loaded dataset:", df.shape)

# ✅ Add expense ratio (optional but important feature)
df["expense_ratio"] = df["monthly_expenses"] / df["monthly_income"]

# ✅ Features & target
X = df.drop("default", axis=1)
y = df["default"]

# ✅ Categorical & numerical columns
categorical_cols = ["gender", "region"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ✅ Preprocessor (OneHotEncode categorical vars)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# ✅ Model
model = RandomForestClassifier(n_estimators=150, random_state=42)

# ✅ Pipeline (preprocessing + model)
clf = Pipeline(steps=[("preprocessor", preprocessor),
                      ("model", model)])

# ✅ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
clf.fit(X_train, y_train)
print("\n✅ Model trained!")

# ✅ Predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

# ✅ Metrics
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\n📊 Accuracy: {acc:.2f} | ROC-AUC: {roc_auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save trained model
joblib.dump(clf, "model/credit_model.pkl")
print("\n✅ Model saved at model/credit_model.pkl")

# ✅ Feature Importance (from model)
print("\n🔥 Feature Importance:")
feature_names_num = numerical_cols
feature_names_cat = list(clf.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_cols))
feature_names_all = feature_names_cat + feature_names_num

importances = clf.named_steps['model'].feature_importances_
feat_imp = pd.DataFrame({"feature": feature_names_all, "importance": importances})
feat_imp = feat_imp.sort_values(by="importance", ascending=False)
print(feat_imp.head(10))
feat_imp.to_csv("data/feature_importance.csv", index=False)
print("\n✅ Saved feature importance to data/feature_importance.csv")

# ✅ Save metrics
with open("data/model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.2f}\n")
    f.write(f"ROC-AUC: {roc_auc:.2f}\n")
print("\n✅ Saved model metrics to data/model_metrics.txt")

# ✅ Save feature importance CSV
feat_imp.to_csv("data/feature_importance.csv", index=False)
