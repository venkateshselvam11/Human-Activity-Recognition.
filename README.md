# Human-Activity-Recognition.
The application is deployed using Streamlit Community Cloud. Users can access the Human Activity Recognition system through a web browser using the shared application link.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load training and testing data
X_train = pd.read_csv(
    "UCI HAR Dataset/train/X_train.txt",
    sep=r"\s+", header=None
)
y_train = pd.read_csv(
    "UCI HAR Dataset/train/y_train.txt",
    sep=r"\s+", header=None
).values.ravel()

X_test = pd.read_csv(
    "UCI HAR Dataset/test/X_test.txt",
    sep=r"\s+", header=None
)
y_test = pd.read_csv(
    "UCI HAR Dataset/test/y_test.txt",
    sep=r"\s+", header=None
).values.ravel()

# Activity labels
activities = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}

# Train the machine learning model
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# Predict activities
predictions = model.predict(X_test)

# Display results
accuracy = accuracy_score(y_test, predictions)
print("Human Activity Recognition System")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

for i in range(5):
    print("Actual Activity:", activities[y_test[i]])
    print("Predicted Activity:", activities[predictions[i]])
    print()
