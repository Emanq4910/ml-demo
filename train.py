# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Dataset load
print("Loading dataset...")
data = load_iris()

# Step 2: Split data into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Step 3: Train the model
print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Step 5: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 6: Show accuracy in logs
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Optional: Save model (if needed for later use)
import joblib
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
