import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("knn-dataset.csv")

# Features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

# Predict test set results
y_pred = knn.predict(X_test)

# Accuracy and confusion matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Predicted values:")
print(y_pred)

print("\nActual values:")
print(y_test)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)
