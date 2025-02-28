# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 2: Read the CSV file
df = pd.read_csv('bank.csv', delimiter=';')

# Step 3: Inspect the dataframe
print(df.head())  # Check the first few rows
print(df.info())  # Check column names and data types

# Step 4: Select specific columns for the second dataframe
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 5: Convert categorical variables to dummy numerical values
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

# Convert the 'y' column to numeric values (1 for 'yes', 0 for 'no')
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

# Step 6: Produce a heat map of correlation coefficients
plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 7: Select target variable y and explanatory variables X
X = df3.drop('y', axis=1)
y = df3['y']

# Step 8: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 9: Setup and train a logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Step 10: Print confusion matrix and accuracy score for logistic regression
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
accuracy_log = accuracy_score(y_test, y_pred_log)
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_log)
print(f"Accuracy Score for Logistic Regression: {accuracy_log:.2f}")

# Step 11: Setup and train a k-nearest neighbors model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Step 12: Print confusion matrix and accuracy score for k-nearest neighbors
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Confusion Matrix for K-Nearest Neighbors:")
print(conf_matrix_knn)
print(f"Accuracy Score for K-Nearest Neighbors: {accuracy_knn:.2f}")

# Step 13: Compare the results between the two models
print("\nComparison of Models:")
print(f"Logistic Regression Accuracy: {accuracy_log:.2f}")
print(f"K-Nearest Neighbors Accuracy: {accuracy_knn:.2f}")