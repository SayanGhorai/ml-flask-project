import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dummy dataset
data = {
    "feature1": [5.1, 4.9, 4.7, 6.4, 6.5],
    "feature2": [3.5, 3.0, 3.2, 3.2, 2.8],
    "feature3": [1.4, 1.4, 1.3, 5.3, 4.6],
    "feature4": [0.2, 0.2, 0.2, 2.3, 1.5],
    "target": [0, 0, 0, 1, 1],
}
df = pd.DataFrame(data)

# Split dataset
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
