import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import os

# 🔹 Path to your combined dataset
input_path = r"C:\Users\Jeyabala\Downloads\EV_d.csv"

# 🔹 Read the dataset
df = pd.read_csv(input_path)
print("✅ Loaded dataset shape:", df.shape)

# ======================================================
# 🧹 STEP 1 — Remove duplicates & reset index
# ======================================================
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print("✅ Duplicates removed. Shape:", df.shape)

# ======================================================
# 🧩 STEP 2 — Handle missing values
# ======================================================
# Numeric columns: fill with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Non-numeric columns: fill with mode
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

print("✅ Missing values handled")

# ======================================================
# 🕓 STEP 3 — Convert datetime column
# ======================================================
if 'start_time' in df.columns:
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['hour'] = df['start_time'].dt.hour
    df['day'] = df['start_time'].dt.day
    df['month'] = df['start_time'].dt.month
    df['year'] = df['start_time'].dt.year
    print("✅ Converted 'start_time' to datetime features")

# ======================================================
# 🧮 STEP 4 — Encode categorical columns
# ======================================================
cat_cols = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

print("✅ Encoded categorical columns")

# ======================================================
# 📏 STEP 5 — Normalize numeric columns (0–1 scaling)
# ======================================================
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("✅ Normalized numeric columns to range [0, 1]")

# ======================================================
# 🧠 STEP 6 — Save preprocessed dataset
# ======================================================
output_path = os.path.join(os.path.dirname(input_path), "Preprocessed_Battery_Dataset.csv")
df.to_csv(output_path, index=False)

print("\n🎯 Preprocessing complete!")
print(f"📁 Saved clean dataset at: {output_path}")
print(f"📊 Final shape: {df.shape}")
