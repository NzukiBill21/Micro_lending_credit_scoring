import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ✅ Load model
model = joblib.load("model/credit_model.pkl")

# ✅ Load model metrics
try:
    with open("data/model_metrics.txt") as f:
        metrics = f.readlines()
    accuracy = metrics[0].split(":")[1].strip()
    roc_auc = metrics[1].split(":")[1].strip()
except:
    accuracy = "N/A"
    roc_auc = "N/A"

# ✅ Load feature importance
try:
    feat_imp = pd.read_csv("data/feature_importance.csv")
except:
    feat_imp = None

# ✅ Streamlit page config
st.set_page_config(page_title="Micro-Lending Credit Scoring", layout="wide")

st.title("💳 Micro-Lending Credit Scoring Dashboard")

# ✅ Show model KPIs
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("✅ Accuracy", accuracy)
col2.metric("📈 ROC-AUC", roc_auc)

st.markdown("---")

# ✅ Feature Importance Visualization
if feat_imp is not None:
    st.subheader("🔥 Top 10 Most Important Features")
    fig = px.bar(
        feat_imp.head(10).sort_values(by="importance", ascending=True),
        x="importance", y="feature",
        orientation="h",
        title="Feature Importance (Random Forest)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Feature importance not available yet. Train the model first.")

st.markdown("---")

# ✅ Borrower Prediction Form
st.subheader("📝 Predict Borrower Default Risk")

age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
region = st.selectbox("Region", ["Nairobi", "Mombasa", "Kisumu", "Rural"])
monthly_income = st.number_input("Monthly Income (KES)", min_value=5000, max_value=100000, value=30000)
monthly_expenses = st.number_input("Monthly Expenses (KES)", min_value=1000, max_value=90000, value=20000)
transaction_count = st.slider("Transaction Count (last 6 months)", 5, 80, 20)
loan_amount = st.number_input("Requested Loan Amount", min_value=2000, max_value=50000, value=10000)
num_loans = st.slider("Number of Previous Loans", 0, 10, 2)
prev_defaults = st.slider("Previous Defaults", 0, 4, 0)

if st.button("Predict Default Risk"):
    # Create input dataframe
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "region": region,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "transaction_count": transaction_count,
        "loan_amount": loan_amount,
        "num_loans": num_loans,
        "prev_defaults": prev_defaults,
        "expense_ratio": monthly_expenses / monthly_income
    }])

    prob_default = model.predict_proba(input_df)[:, 1][0]
    pred_default = model.predict(input_df)[0]

    st.subheader(f"Default Risk Probability: **{prob_default:.2%}**")
    if pred_default == 1:
        st.error("⚠️ High risk borrower!")
    else:
        st.success("✅ Low risk borrower")

st.write("---")
st.caption("Powered by Streamlit + RandomForest ML model")
