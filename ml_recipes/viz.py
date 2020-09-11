import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris() 
test_idx = [0, 50, 100]

# training data
train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx)

# testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# train the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print("the meta data: features and lables")
print(iris.feature_names)
print(iris.target_names)
print("the testing data & target and the prediction from the classifier")
print(test_data, test_target)
print(clf.predict(test_data))
print("the testing data of index 1 to show mechanism manually")
print(test_data[1])
print(test_target[1])

from sklearn.tree import export_graphviz
with open(r".\tree.dot", 'w') as f:
    export_graphviz(clf,
                     out_file=f,
                     feature_names=iris.feature_names[:],
                     class_names=iris.target_names,
                     rounded=True,
                     filled=True)

#from graphviz import Source
#dot_path = "./tree.dot"
#output = Souce.from_file(dot_path, format = "png") # can change png to pdf
#output.view()