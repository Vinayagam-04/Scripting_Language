import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("C:/Users/vinay/OneDrive/Desktop/diabetes/diabetes.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Check class balance
print("Class distribution before SMOTE:\n", y.value_counts())

# Apply SMOTE only if imbalance exists
if y.value_counts()[0] != y.value_counts()[1]:
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    print("Class distribution after SMOTE:\n", y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)        # Transform test data

# Define Random Forest model with hyperparameter tuning
rf_model = RandomForestClassifier()
rf_params = {
    'n_estimators': [50, 100],  
    'max_depth': [10, None],  
    'criterion': ['gini']  
}

# GridSearch for best parameters
grid_search_rf = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)

# Best model from GridSearch
best_rf_model = grid_search_rf.best_estimator_

# Predictions
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Model Evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Best Accuracy: {rf_accuracy:.4f}")
print(f"Best Hyperparameters: {grid_search_rf.best_params_}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Function to take user input and predict
def get_user_input():
    print("\nEnter details for Diabetes Prediction:\n")
    input_data = {
        "Pregnancies": float(input("Enter number of Pregnancies: ")),
        "Glucose": float(input("Enter Glucose level: ")),
        "BloodPressure": float(input("Enter Blood Pressure: ")),
        "SkinThickness": float(input("Enter Skin Thickness: ")),
        "Insulin": float(input("Enter Insulin level: ")),
        "BMI": float(input("Enter BMI: ")),
        "DiabetesPedigreeFunction": float(input("Enter Diabetes Pedigree Function: ")),
        "Age": float(input("Enter Age: "))
    }

    # Convert input into a DataFrame with column names
    user_df = pd.DataFrame([input_data])

    # Scale user input using the same StandardScaler
    user_data_scaled = scaler.transform(user_df)  # Ensure 2D

    return user_data_scaled

# Get user input and make a prediction
user_input = get_user_input()
user_prediction = best_rf_model.predict(user_input)

# Display result
if user_prediction[0] == 1:
    print("\nThe model predicts: **Diabetic** 🩸")
else:
    print("\nThe model predicts: **Non-Diabetic** ✅")
