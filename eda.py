# =========================================================
# ⚡ EDA for Battery Dataset
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1️⃣ Load the dataset
# ---------------------------------------------------------
df = pd.read_csv(r"C:\Users\Jeyabala\Downloads\EV_d.csv")
print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("\n🔍 Columns:\n", df.columns.tolist())

# ---------------------------------------------------------
# 2️⃣ Basic Information
# ---------------------------------------------------------
print("\n📘 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe())

# ---------------------------------------------------------
# 3️⃣ Missing Values Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# ---------------------------------------------------------
# 4️⃣ Correlation Heatmap (for numeric columns)
# ---------------------------------------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ---------------------------------------------------------
# 5️⃣ Distribution Plot of Numeric Columns
# ---------------------------------------------------------
numeric_cols = df.select_dtypes(include=['float64','int64']).columns

plt.figure(figsize=(15,10))
for i, col in enumerate(numeric_cols[:6]):  # first 6 numeric features
    plt.subplot(2,3,i+1)
    sns.histplot(df[col], kde=True, bins=30, color='teal')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6️⃣ Pairplot for Relationships Between Key Features
# ---------------------------------------------------------
sample_cols = ['Sense_current','Battery_current','Voltage_charge','Capacity','Re','Rct']
sample_cols = [col for col in sample_cols if col in df.columns]
if len(sample_cols) >= 3:
    sns.pairplot(df[sample_cols], diag_kind='kde')
    plt.suptitle("Feature Relationship Pairplot", y=1.02)
    plt.show()

# ---------------------------------------------------------
# 7️⃣ Boxplot to Detect Outliers in Key Parameters
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df[sample_cols])
plt.title("Outlier Detection - Key Features")
plt.xticks(rotation=45)
plt.show()

# ---------------------------------------------------------
# 8️⃣ Line Plot — Voltage vs Current Over Time
# ---------------------------------------------------------
if 'start_time' in df.columns:
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df = df.sort_values('start_time')
    plt.figure(figsize=(12,5))
    sns.lineplot(x='start_time', y='Voltage_charge', data=df, label='Voltage Charge', color='orange')
    sns.lineplot(x='start_time', y='Battery_current', data=df, label='Battery Current', color='blue')
    plt.title("Voltage & Current Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
