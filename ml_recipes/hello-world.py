from sklearn import tree
# feature : smooth: 0; bumpy: 1; 
# label:    orange: 0; apple: 1;
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))