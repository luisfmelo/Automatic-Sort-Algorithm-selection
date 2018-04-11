from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from _helpers.Database import Database

db = Database()
dataset = db.load_csv('../csv_files/features.csv')
# X = df['length'].values
# y = df['target'].values
X = dataset.iloc[:, [dataset.columns.get_loc("length")]].values
y = dataset.iloc[:, dataset.columns.get_loc("target")].values

# Get Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)

# fit to model
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y_train)

# Predict Output
y_pred = dtree.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Get Accuracy
accuracy = accuracy_score(y_test, y_pred)*100

print("Accuracy with Decision Tree is: {} %".format(accuracy))
