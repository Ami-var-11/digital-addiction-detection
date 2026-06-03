import pandas as pd
import os

print("=" * 60)
print("🔧 PREPARING KAGGLE DATA FOR ML")
print("=" * 60)

# Path to Kaggle data folder
data_folder = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\data\kaggle_data"

# Find the CSV file
csv_files = []
for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        csv_files.append(os.path.join(data_folder, file))

if not csv_files:
    print("❌ No CSV file found!")
    exit()

file_path = csv_files[0]
print(f"✅ Found file: {os.path.basename(file_path)}")

# Load the data
df = pd.read_csv(file_path)
print(f"✅ Loaded {len(df)} rows")

# Select features for ML
feature_cols = ['daily_screen_time_min', 'num_app_switches', 'social_media_time_min']
X = df[feature_cols].copy()

# Use anxiety_level as label (1-10 scale)
# We'll consider >7 as "addicted" (1) and <=7 as "normal" (0)
y = (df['anxiety_level'] > 7).astype(int)

print(f"\n📊 Features selected: {feature_cols}")
print(f"📊 Label distribution:")
print(f"   Normal (0): {sum(y==0)} rows")
print(f"   Addicted (1): {sum(y==1)} rows")

# Check for missing values
print(f"\n🔍 Missing values:")
print(X.isnull().sum())

# Fill missing values if any
X = X.fillna(0)

# Combine features and label
prepared_df = X.copy()
prepared_df['label'] = y

# Save prepared data
output_path = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\data\kaggle_prepared.csv"
prepared_df.to_csv(output_path, index=False)

print(f"\n✅ Prepared data saved to: {output_path}")
print(f"📊 Final shape: {prepared_df.shape}")
print(f"\n📋 First 5 rows of prepared data:")
print(prepared_df.head())