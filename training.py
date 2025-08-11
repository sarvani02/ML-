import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
data = pd.read_csv("student_records.csv")

# One-hot encode categorical columns
onehot_cols = ["OverallGrade", "Obedient"]
onehot = OneHotEncoder(drop="first", sparse_output=False)
onehot_df = pd.DataFrame(onehot.fit_transform(data[onehot_cols]),
                         columns=onehot.get_feature_names_out(onehot_cols),
                         index=data.index)

# Drop unused columns
data = data.drop(columns=onehot_cols + ["Name"])
data = pd.concat([data, onehot_df], axis=1)

# Label encode target
label = LabelEncoder()
data['Recommend'] = label.fit_transform(data['Recommend'])

# Split features and target
y = data['Recommend']
X = data.drop(columns='Recommend')

# Balance data
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label.pkl", "wb") as f:
    pickle.dump(label, f)
with open("onehot.pkl", "wb") as f:
    pickle.dump(onehot, f)

# Evaluate
print("Model accuracy:", model.score(X_test, y_test))
