# 💳 Micro-Lending Credit Scoring  

This is a **real-world ML project** that predicts **borrower default risk** for micro-lending institutions.  

 **Features:**  
- Generates synthetic borrower data (income, expenses, loan history)  
- Exploratory Data Analysis (EDA)  
- Trains a RandomForest model for credit scoring  
- Shows feature importance  
- Interactive **Streamlit app** for risk prediction  

---

## 📊 Model Performance  
- Accuracy: ~85-90%  
- ROC-AUC: ~0.9  
- Most important features: `prev_defaults`, `expense_ratio`, `monthly_income`  

---

## 🚀 How to Run Locally  

```bash
# 1️⃣ Clone repo
git clone https://github.com/NzukiBill/Micro_lending_credit_scoring.git
cd Micro_lending_credit_scoring

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Train model (optional)
python train_model.py

# 4️⃣ Run app
streamlit run app/app.py
