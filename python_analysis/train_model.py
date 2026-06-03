import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("🤖 TRAINING RANDOM FOREST MODEL")
print("=" * 60)

# Load prepared data
data_path = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\data\kaggle_prepared.csv"
df = pd.read_csv(data_path)

print(f"\n✅ Loaded {len(df)} samples")

# Separate features and label
feature_cols = ['daily_screen_time_min', 'num_app_switches', 'social_media_time_min']
X = df[feature_cols]
y = df['label']

print(f"\n📊 Features: {feature_cols}")
print(f"📊 Training samples: {len(X)}")
print(f"📊 Class distribution:")
print(f"   Normal (0): {sum(y==0)} samples")
print(f"   Addicted (1): {sum(y==1)} samples")

# Split data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Train set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,           # Maximum depth of trees
    min_samples_split=5,    # Minimum samples to split a node
    random_state=42
)

print("\n⏳ Training model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test accuracy: {accuracy:.2%}")

# Detailed classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Addicted'],
            yticklabels=['Normal', 'Addicted'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

# Save confusion matrix
models_dir = r"C:\Users\suved_8hu\OneDrive\Desktop\DigitalAddictionProject\models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
plt.show()

# Cross-validation (more reliable accuracy estimate)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\n📊 Cross-validation scores: {cv_scores}")
print(f"📊 Average CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Feature Importance:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Addiction Detection')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'feature_importance.png'))
plt.show()

# Save the model
model_path = os.path.join(models_dir, 'addiction_model.pkl')
joblib.dump(model, model_path)
print(f"\n✅ Model saved to: {model_path}")

# Save feature names for later use
with open(os.path.join(models_dir, 'feature_names.txt'), 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

print("✅ Feature names saved")

# Quick test with sample data
print("\n" + "=" * 60)
print("🔮 TESTING MODEL WITH SAMPLE DATA")
print("=" * 60)

# Create some sample test cases
test_cases = [
    [120, 10, 30],   # Low usage (should be Normal)
    [480, 50, 240],  # High usage (should be Addicted)
    [300, 25, 120],  # Medium usage
]

for i, test in enumerate(test_cases):
    features = np.array([test])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    print(f"\n📱 Test Case {i+1}:")
    print(f"   Screen time: {test[0]} mins ({test[0]/60:.1f} hours)")
    print(f"   App switches: {test[1]}")
    print(f"   Social media: {test[2]} mins")
    print(f"   Prediction: {'ADDICTED' if pred == 1 else 'NORMAL'}")
    print(f"   Confidence: {max(prob):.1%}")