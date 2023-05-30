import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm

if len(sys.argv) < 2:
    print("Requires argument to find data: either 1, 2 or 4")
    sys.exit(1)

data_fol = sys.argv[1]

if(data_fol == '1'):
    train1Directory = 'project1_datasets/enron1_train/enron1/train'
    test1Directory = 'project1_datasets/enron1_test/enron1/test'

elif(data_fol == '2'):
    train1Directory = 'project1_datasets/enron2_train/train'
    test1Directory = 'project1_datasets/enron2_test/test'

elif(data_fol == '4'): 
    train1Directory = 'project1_datasets/enron4_train/enron4/train'
    test1Directory = 'project1_datasets/enron4_test/enron4/test'

ham1Train = train1Directory + '/ham'
spam1Train = train1Directory + '/spam'
ham1Test = test1Directory + '/ham'
spam1Test = test1Directory + '/spam'

#making bag of words and bernoulli model for ham and spam
hams = []
spams = []

for file in os.listdir(ham1Train):
    f = os.path.join(ham1Train, file)
    if os.path.isfile(f):
        email = open(f, 'r', errors = 'ignore')
        hams.append(email.read())
    else:
        print("Training file could not be found")
        sys.exit()



for file in os.listdir(spam1Train):
    f = os.path.join(spam1Train, file)
    if os.path.isfile(f):
        email = open(f, 'r', errors = 'ignore')
        spams.append(email.read())
    else:
        print("Training file could not be found")
        sys.exit()

#creating test data

testData = []
testClasses = []


for file in os.listdir(ham1Test):
    fileLoc = os.path.join(ham1Test, file)
    email = open(fileLoc, 'r', errors = 'ignore')
    testData.append(email.read())
    testClasses.append(1)

for file in os.listdir(spam1Test):
    fileLoc = os.path.join(spam1Test, file)
    email = open(fileLoc, 'r', errors = 'ignore')
    testData.append(email.read())
    testClasses.append(0)


#creating SGDClassifier

hamclasses = [1] * len(hams)

spamclasses = [0] * len(spams)

classes = np.concatenate([hamclasses, spamclasses])

trainData = np.concatenate((hams, spams))

trainData = np.concatenate((trainData, testData))

classes = np.concatenate((classes, testClasses))

bow_clf = Pipeline([('vect', CountVectorizer()), ('sgdcBow', SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, max_iter=15, tol=None)),])

X_train, X_test, y_train, y_test = train_test_split(trainData, classes)

bow_clf.fit(X_train, y_train)

bow_pred = bow_clf.predict(X_test)

#trvec = CountVectorizer()
#ber_Xtrain = trvec.fit_transform(X_train)
#ber_Xtrain[ber_Xtrain>0] = 1

#sgdcBer = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,     max_iter=5, tol=None)

#sgdcBer.fit(ber_Xtrain, y_train)

#ber_pred = sgdcBer.predict(X_test)

#print results

print("Bag of Words = \nAccuracy: " + str(skm.accuracy_score(y_test, bow_pred)) + "\nPrecision: " + str(skm.precision_score(y_test, bow_pred)) + "\nRecall: " + str(skm.recall_score(y_test, bow_pred)) + "\nF1 Score: " + str(skm.f1_score(y_test, bow_pred)) + "\n")
#print("Bernoulli = \nAccuracy: " + str(skm.accuracy_score(y_test, ber_pred)) + "\nPrecision: " + str(skm.precision_score(y_test, ber_pred)) + "\nRecall: " + str(skm.recall_score(y_test, ber_pred)) + "\nF1 Score: " + str(skm.f1_score(y_test, ber_pred)) + "\n")