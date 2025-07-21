import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load dataset
df = pd.read_csv("data/synthetic_microlending.csv")
print("✅ Dataset loaded with shape:", df.shape)

# ✅ Basic info
print("\n--- Basic Info ---")
print(df.info())
print("\n--- Missing values ---")
print(df.isnull().sum())

# ✅ Quick stats
print("\n--- Summary Statistics ---")
print(df.describe())

# ✅ Default rate
default_rate = df["default"].mean()
print(f"\nDefault Rate: {default_rate:.2%}")

# ✅ Plot default distribution
sns.countplot(x="default", data=df)
plt.title("Default vs Non-Default Counts")
plt.savefig("data/default_distribution.png")
plt.close()

# ✅ Correlation heatmap (numeric only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # select only numeric columns
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("data/correlation_heatmap.png")
plt.close()

# ✅ Income vs Default
plt.figure(figsize=(8,5))
sns.boxplot(x="default", y="monthly_income", data=df)
plt.title("Monthly Income vs Default")
plt.savefig("data/income_vs_default.png")
plt.close()

# ✅ Expense ratio analysis
df["expense_ratio"] = df["monthly_expenses"] / df["monthly_income"]
plt.figure(figsize=(8,5))
sns.boxplot(x="default", y="expense_ratio", data=df)
plt.title("Expense Ratio vs Default")
plt.savefig("data/expense_ratio_vs_default.png")
plt.close()

print("\n✅ EDA done! Check generated plots in the data/ folder.")
