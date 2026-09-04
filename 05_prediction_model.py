"""
Step 5: Basic Crime Prediction Model
Crime Pattern Detection System

Goal: Given details like City, Time of Day, Hour, Month -> 
predict whether the crime will be "Violent Crime" or not.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned data
df = pd.read_pickle('cleaned_crime_data.pkl')

# Create target variable: 1 if Violent Crime, 0 otherwise
df['Is_Violent'] = (df['Crime Domain'] == 'Violent Crime').astype(int)

# Select features we'll use to predict
features = ['City', 'Time_Bucket', 'Occurrence_Hour', 'Occurrence_MonthNum',
            'Victim Gender', 'Victim Age']
df_model = df[features + ['Is_Violent']].dropna()

# Encode categorical (text) columns into numbers, since ML models need numbers
label_encoders = {}
for col in ['City', 'Time_Bucket', 'Victim Gender']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Split data into inputs (X) and target (y)
X = df_model[features]
y = df_model['Is_Violent']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("=" * 60)
print()
print(classification_report(y_test, y_pred, target_names=['Not Violent', 'Violent']))

# Which features matter most for prediction?
print("=" * 60)
print("FEATURE IMPORTANCE (what matters most for prediction)")
print("=" * 60)
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importance)