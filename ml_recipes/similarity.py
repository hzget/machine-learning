##########################################################
# the following is the class that implement the 'similarity' classifier
# we just need to implement the interfaces fit() and predict()
from scipy.spatial import distance
def euc(a, b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test :
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    # a helper function that get the label of the nearest training data
    def closest(self, row):
        dist = euc(row, self.X_train[0])
        idx = 0
        for i in range(1,len(self.X_train)) :
            d = euc(row, self.X_train[i])
            if d < dist :
                dist = d
                idx = i
        return self.y_train[idx]

##########################################################
# the following is how to use the new classifier to resolve issues:
from sklearn import datasets
iris = datasets.load_iris()

# a classifier is a function f(X) = y
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# multiple classifiers can resolve the same problems
# just only need to replace the factory method. 
# Example 1:
#     from sklearn import tree
#     my_classifier = tree.DecisionTreeClassifier()
# Example 2:
#     from sklearn.neighbors import KNeighborsClassifier
#     my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

# different classifiers have the same interfaces
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

# verify the accuracy with testing data
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))