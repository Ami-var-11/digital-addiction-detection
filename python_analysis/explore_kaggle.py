import pandas as pd
import os

print("=" * 60)
print("📊 KAGGLE DATASET EXPLORATION")
print("=" * 60)

# Path to your Kaggle data folder
data_folder = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\data\kaggle_data"

print(f"📁 Looking in: {data_folder}")

# Check if folder exists
if not os.path.exists(data_folder):
    print(f"❌ Folder not found: {data_folder}")
    exit()

# Find ALL data files (CSV or Excel)
data_files = []
for file in os.listdir(data_folder):
    if file.endswith('.csv') or file.endswith('.xlsx') or file.endswith('.xls'):
        data_files.append(os.path.join(data_folder, file))

if not data_files:
    print("❌ No data files found in the folder!")
    print("Please place your Kaggle dataset file in:")
    print(data_folder)
    exit()

print(f"✅ Found {len(data_files)} file(s):")
for f in data_files:
    print(f"   - {os.path.basename(f)}")

# Use the first file
file_path = data_files[0]
print(f"\n📂 Using file: {os.path.basename(file_path)}")

# Load based on file extension
try:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        print("✅ CSV file loaded successfully!")
    else:
        df = pd.read_excel(file_path)
        print("✅ Excel file loaded successfully!")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

print(f"\n📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📋 Column names:")
for i, col in enumerate(df.columns):
    print(f"   {i+1}. {col}")

print(f"\n📊 First 5 rows:")
print(df.head())

print(f"\n📊 Data types:")
print(df.dtypes)

print(f"\n📊 Missing values:")
print(df.isnull().sum())

# Save column info
info_path = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\data\column_info.txt"
with open(info_path, "w") as f:
    f.write("KAGGLE DATASET COLUMNS:\n")
    f.write("=" * 50 + "\n")
    for i, col in enumerate(df.columns):
        f.write(f"{i+1}. {col}\n")

print(f"\n✅ Column information saved to column_info.txt")