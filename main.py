from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Define the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Define the hyperparameters to search over
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

# Use GridSearchCV to search for the best hyperparameters
search = GridSearchCV(clf, param_grid, cv=5)
search.fit(X_train, y_train)

# Get the best hyperparameters and fit the model on the training set
best_params = search.best_params_
clf = DecisionTreeClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set and print classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



# visualize the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
plt.show()
