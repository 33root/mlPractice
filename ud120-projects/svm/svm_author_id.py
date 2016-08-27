#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
#from class_vis import prettyPicture

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score
import time

#slice the dataset to 1% of its size
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

gnb = svm.SVC(kernel='rbf', C=10000)

t0 = time.time()
gnb.fit(features_train, labels_train)
print "training time:", round(time.time() - t0,3), "s"

t1 = time.time()
pred = gnb.predict(features_test)
print "prediction time:", round(time.time() - t1,3), "s"
print "Accuracy:", accuracy_score(pred, labels_test)
#prettyPicture(gnb, features_test, labels_test)
#answer10 = pred[10]
#answer26 = pred[26]
#answer50 = pred[50]

n = 0

for predictions in pred:
	if predictions == 1:
		n = n+1

#print "elemento 10:", answer10
#print "elemento 26:", answer26
#print "elemento 50:", answer50

print "mails de cris:", n
#########################################################


