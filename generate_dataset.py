import numpy as np
import pandas as pd
import random

np.random.seed(42)

def generate_borrower_data(n=5000):
    data = []
    regions = ["Nairobi", "Mombasa", "Kisumu", "Rural"]
    genders = ["Male", "Female"]

    for _ in range(n):
        age = np.random.randint(18, 65)
        gender = random.choice(genders)
        region = random.choice(regions)

        monthly_income = np.random.randint(5000, 100000)
        monthly_expenses = monthly_income * np.random.uniform(0.4, 0.9)
        transaction_count = np.random.randint(5, 80)
        loan_amount = np.random.randint(2000, 50000)
        num_loans = np.random.randint(0, 10)
        prev_defaults = np.random.randint(0, min(num_loans+1, 5))

        expense_ratio = monthly_expenses / monthly_income

        default_prob = (
            0.3 * (expense_ratio > 0.8) +
            0.3 * (prev_defaults > 1) +
            0.2 * (monthly_income < 15000) +
            0.2 * (loan_amount > monthly_income * 0.8)
        )

        default = 1 if random.random() < default_prob else 0

        data.append([
            age, gender, region, monthly_income, monthly_expenses,
            transaction_count, loan_amount, num_loans, prev_defaults, default
        ])

    columns = [
        "age", "gender", "region", "monthly_income", "monthly_expenses",
        "transaction_count", "loan_amount", "num_loans", "prev_defaults", "default"
    ]
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    df = generate_borrower_data(5000)
    df.to_csv("synthetic_microlending.csv", index=False)
    print("✅ Dataset generated: synthetic_microlending.csv")
    print(df.head())
