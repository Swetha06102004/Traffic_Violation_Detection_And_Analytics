import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

new_df = pd.read_csv("new_data.csv")

new_df['Fine_Paid'] = new_df['Fine_Paid'].astype(str).str.strip().replace({'no': 'No', 'yes': 'Yes'})
new_df['Vehicle_Type'] = new_df['Vehicle_Type'].astype(str).str.strip().replace({'private': 'Private', 'commercial': 'Commercial'})
new_df['Violation_Type'] = new_df['Violation_Type'].astype(str).str.strip().replace({'speeding': 'Speeding', 'signal jump': 'Signal Jump'})

le_fine_paid = joblib.load("le_fine_paid.pkl")
le_vehicle_type = joblib.load("le_vehicle_type.pkl")
le_violation_type = joblib.load("le_violation_type.pkl")

valid_fine_paid = list(le_fine_paid.classes_)
valid_vehicle_type = list(le_vehicle_type.classes_)
valid_violation_type = list(le_violation_type.classes_)

if not set(new_df['Fine_Paid']).issubset(valid_fine_paid):
    new_df.loc[~new_df['Fine_Paid'].isin(valid_fine_paid), 'Fine_Paid'] = valid_fine_paid[0]

if not set(new_df['Vehicle_Type']).issubset(valid_vehicle_type):
    new_df.loc[~new_df['Vehicle_Type'].isin(valid_vehicle_type), 'Vehicle_Type'] = valid_vehicle_type[0]

if not set(new_df['Violation_Type']).issubset(valid_violation_type):
    new_df.loc[~new_df['Violation_Type'].isin(valid_violation_type), 'Violation_Type'] = valid_violation_type[0]

new_df['Fine_Paid'] = le_fine_paid.transform(new_df['Fine_Paid'])
new_df['Vehicle_Type'] = le_vehicle_type.transform(new_df['Vehicle_Type'])
new_df['Violation_Type'] = le_violation_type.transform(new_df['Violation_Type'])

features = ['Violation_Type', 'Fine_Amount', 'Speed_Limit','Recorded_Speed', 'Alcohol_Level', 'Driver_Age','Vehicle_Type', 'Previous_Violations', 'Fine_Paid']

for col in features:
    if col not in new_df.columns:
        print(f"Column '{col}' missing. Filling with 0.")
        new_df[col] = 0

X_unseen = new_df[features]

X_unseen = X_unseen.fillna(0)

model_path = "fraud_detection_model.pkl"

if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found. Exiting prediction.")
    sys.exit(1)

try:
    model = joblib.load(model_path)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

if X_unseen.empty:
    print("No valid data available to predict.")
else:
    new_df['Predicted_Is_Fraudulent'] = model.predict(X_unseen)

    new_df['Fraud_Prob'] = model.predict_proba(X_unseen)[:, 1]
    new_df.to_csv("predicted_fraud_results.csv", index=False)
    print("Predictions saved to predicted_fraud_results.csv")