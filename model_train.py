import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("symptom_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("✅ Columns Loaded:", df.columns.tolist())
print("🔹 Dataset shape before cleaning:", df.shape)

# Drop unnecessary or text columns
drop_cols = ['PatientID', 'DoctorInCharge', 'Ethnicity', 'EducationLevel']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Replace Yes/No with 1/0
df = df.replace({'Yes': 1, 'No': 0, 'Mild': 1, 'Severe': 1, 'None': 0, 'Present': 1, 'Absent': 0})

# Convert Gender to numeric if not already
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})

# Handle non-numeric issues safely (but do NOT drop all)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with column mean instead of dropping rows
df = df.fillna(df.mean(numeric_only=True))

print("🔹 Dataset shape after cleaning:", df.shape)

# Separate features & target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Encode target if text
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Sanity check
print("✅ Final dataset ready for training.")
print("Feature columns:", X.columns.tolist())
print("Target unique values:", set(y))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# Save
with open('dementia_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("💾 Model saved as dementia_model.pkl")
