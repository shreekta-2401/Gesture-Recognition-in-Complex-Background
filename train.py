import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv('up_gestures.csv')
data_down = pd.read_csv('down_gestures.csv')
data = pd.concat([data, data_down], ignore_index=True)

# Prepare data
X = data.drop('gesture', axis=1)
y = data['gesture']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, 'gesture_model.pkl')
