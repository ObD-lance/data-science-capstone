import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(y, y_predict):
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()

# Load the data
data = pd.read_csv("dataset_part_2.csv")
X = pd.read_csv("dataset_part_3.csv")

# Create a NumPy array from the column Class in data, by applying the method to_numpy()
# then assign it  to the variable Y, make sure the output is a Pandas series (only one bracket df['name of  column']).
Y = data['Class'].to_numpy()

# TASK 2: Standardize the data in X
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# TASK 3: Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(Y_test.shape)

# TASK 4: Create and train Logistic Regression model
parameters = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)

print("Logistic Regression tuned hyperparameters :(best parameters) ", logreg_cv.best_params_)
print("Logistic Regression accuracy :", logreg_cv.best_score_)

# TASK 5: Calculate accuracy on test data for Logistic Regression
logreg_accuracy = logreg_cv.score(X_test, Y_test)
print("Logistic Regression Test Accuracy:", logreg_accuracy)

# Plot confusion matrix for Logistic Regression
yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# TASK 6: Create and train SVM model
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma': np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)

print("SVM tuned hyperparameters :(best parameters) ", svm_cv.best_params_)
print("SVM accuracy :", svm_cv.best_score_)

# TASK 7: Calculate accuracy on test data for SVM
svm_accuracy = svm_cv.score(X_test, Y_test)
print("SVM Test Accuracy:", svm_accuracy)

# Plot confusion matrix for SVM
yhat = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)




# TASK 8: Create and train Decision Tree model
parameters = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2*n for n in range(1, 10)],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)

print("Decision Tree tuned hyperparameters :(best parameters) ", tree_cv.best_params_)
print("Decision Tree accuracy :", tree_cv.best_score_)

# TASK 9: Calculate accuracy on test data for Decision Tree
tree_accuracy = tree_cv.score(X_test, Y_test)
print("Decision Tree Test Accuracy:", tree_accuracy)

# Plot confusion matrix for Decision Tree
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)




# TASK 10: Create and train K-Nearest Neighbors model
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, parameters, cv=10)
knn_cv.fit(X_train, Y_train)

print("KNN tuned hyperparameters :(best parameters) ", knn_cv.best_params_)
print("KNN accuracy :", knn_cv.best_score_)

# TASK 11: Calculate accuracy on test data for KNN
knn_accuracy = knn_cv.score(X_test, Y_test)
print("KNN Test Accuracy:", knn_accuracy)

# Plot confusion matrix for KNN
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# TASK 12: Find the method that performs best
models = {
    "Logistic Regression": logreg_accuracy,
    "SVM": svm_accuracy,
    "Decision Tree": tree_accuracy,
    "K-Nearest Neighbors": knn_accuracy
}

best_model = max(models, key=models.get)
print(f"The best performing model is: {best_model} with an accuracy of {models[best_model]:.4f}")

# Visualization of results
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), models.values())
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(models.values()):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.show()