import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sklearn

df = pd.read_csv("../datasets/Indian_Traffic_Violations.csv")

df['Helmet_Worn'].fillna('Unknown', inplace=True)
df['Seatbelt_Worn'].fillna('Unknown', inplace=True)
df['Comments'].fillna('No Comment', inplace=True)

df['OverSpeed'] = df['Recorded_Speed'] > df['Speed_Limit']
df['Alcoholic_Driver'] = df['Alcohol_Level'] > 0.05
df['Is_Fraudulent'] = ((df['OverSpeed'] & df['Alcoholic_Driver']) | ((df['Previous_Violations'] > 3) & (df['Fine_Paid'] == 'No'))).astype(int)

le_fine_paid = LabelEncoder()
df['Fine_Paid'] = le_fine_paid.fit_transform(df['Fine_Paid'])
le_vehicle_type = LabelEncoder()
df['Vehicle_Type'] = le_vehicle_type.fit_transform(df['Vehicle_Type'])
le_violation_type = LabelEncoder()
df['Violation_Type'] = le_violation_type.fit_transform(df['Violation_Type'])

features = ['Violation_Type', 'Fine_Amount', 'Speed_Limit','Recorded_Speed', 'Alcohol_Level', 'Driver_Age','Vehicle_Type', 'Previous_Violations', 'Fine_Paid']

X = df[features].copy()
y = df['Is_Fraudulent'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("✅ Accuracy on test data:", accuracy_score(y_test, y_pred))
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

scores = cross_val_score(model, X, y, cv=5)

print("\n🔁 Cross-validation scores:", scores)
print("📈 Mean CV accuracy:", scores.mean())

X_test = X_test.copy()

X_test['Predicted'] = y_pred
X_test['Actual'] = y_test.values

mismatches = X_test[X_test['Predicted'] != X_test['Actual']]

print(f"\n❗ Mismatches between predicted and actual: {len(mismatches)}")
print("\n📋 Full prediction results (Actual vs Predicted):")
print(X_test[['Predicted', 'Actual']].head(10))

joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(le_violation_type, "le_violation_type.pkl")
joblib.dump(le_vehicle_type, "le_vehicle_type.pkl")
joblib.dump(le_fine_paid, "le_fine_paid.pkl")