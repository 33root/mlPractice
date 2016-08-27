#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
import time
from sklearn.metrics import accuracy_score
#K Nearest Neighbors
print "-----------------K Nearest Neighbors--------------------"
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=9)
t0 = time.time()

clf.fit(features_train, labels_train)
print "training time of k nearest neighbors: ", round( time.time() - t0,3), "s"
t1 = time.time()
pred = clf.predict(features_test)
print "predict time of k nearest neighbors: ", round( time.time() - t1,3), "s"

acc = accuracy_score(pred, labels_test)

print "Accuracy of K Nearest Neighbors: ", acc

print "------------------Random Forest Classifier -------------------"

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15)
t0 = time.time()
clf.fit(features_train, labels_train)
print "Training Time for Random Forest: ", round(time.time() - t0,3), "s"

t1 = time.time()
pred = clf.predict(features_test)
print "Prediction Time for Random Forest: ", round(time.time() - t1,3), "s"

acc = accuracy_score(pred, labels_test)
print "Accuracy of Random Forest:", acc


print "----------------------- AdaBoost ----------------------------"

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

clf = AdaBoostClassifier(base_estimator=SVC(), algorithm='SAMME')
t0 = time.time()
clf.fit(features_train, labels_train)
print "training time for Adaboost: ", round(time.time()-t0,3), "s"

t1 = time.time()
pred = clf.predict(features_test)
print "prediction time for Adaboost: ", round(time.time() - t1,3), "s"

acc = accuracy_score(pred, labels_test)
prettyPicture(clf, features_test, labels_test)
print "Accuracy for AdaBoost: ", acc


