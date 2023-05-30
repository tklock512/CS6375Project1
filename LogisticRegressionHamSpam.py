import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import math
from sklearn.model_selection import train_test_split
import numpy as np

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

#creating Bernoulli/BOW model for hams

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

#creating Bernoulli/BOW model for spams

for file in os.listdir(spam1Train):
    f = os.path.join(spam1Train, file)
    if os.path.isfile(f):
        email = open(f, 'r', errors = 'ignore')
        spams.append(email.read())
    else:
        print("Training file could not be found")
        sys.exit()



hamclasses = [1]*len(hams)
spamclasses = [0]*len(spams)

classes = np.concatenate([hamclasses, spamclasses])

allarray = np.concatenate([hams, spams])

vec = CountVectorizer()
X = vec.fit_transform(allarray)
alldm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

#save words used for the training data
words = alldm.columns

berdm = alldm.copy()
berdm[berdm>0] = 1

#split 30% off for validation
bow_train, bow_validation, bow_train_classes, bow_validation_classes = train_test_split(alldm, classes, test_size=0.30)

ber_train, ber_validation, ber_train_classes, ber_validation_classes = train_test_split(berdm, classes, test_size=0.30)

#return to numpy arrays 

bowFeaturesTrain = bow_train.to_numpy()
bowFeaturesValid = bow_validation.to_numpy()

berFeaturesTrain = ber_train.to_numpy()
berFeaturesValid = ber_validation.to_numpy()

#info for adjusting logistic regression

learning_rate = 0.01
max_iterations = 1000
lambda_val = 10 

initial_weights = np.zeros(bow_train.shape[1])

#function for logistic regression
def logisticRegression(train_data, class_data, initial_weight, l_rate, lamda, max_iter):
    weights = np.array(initial_weight)
    for i in range(max_iter):
        predictions = 1.0/(1+np.exp(-(np.dot(train_data, weights))))
        indicator = (class_data==+1)
        errors = indicator - predictions
        for j in range(len(weights)):
            derivative = np.dot(errors, train_data[:,j])
            if(j!=0):
                derivative = derivative - 2*lamda*weights[j]
            weights[j] = weights[j] + learning_rate*derivative
    return weights

#function for finding accuracy, precision, etc.
def classification_accuracy(train_data, class_data, weights):
    scores = np.dot(train_data, weights)
    threshold = np.vectorize(lambda x: 1. if x > 0 else 0)
    predictions = threshold(scores)
    n_correct = (predictions==class_data).sum()
    n_wrong = len(class_data) - n_correct
    return n_correct, n_wrong



bow_weights = logisticRegression(alldm.to_numpy(), classes, initial_weights, learning_rate, lambda_val, max_iterations)
ber_weights = logisticRegression(berdm.to_numpy(), classes, initial_weights, learning_rate, lambda_val, max_iterations)

#applying model to test data

trueNegative = 0
falsePositive = 0
falseNegative = 0
truePositive = 0

test_data = []


#formatting test data

hamtest = []
spamtest = []

for file in os.listdir(ham1Test):
    f = os.path.join(ham1Test, file)
    if os.path.isfile(f):
        email = open(f, 'r', errors = 'ignore')
        hamtest.append(email.read())
    else:
        print("Testing file could not be found")
        sys.exit()

vec = CountVectorizer(vocabulary=words)
X = vec.transform(hamtest)
hamtestdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

for file in os.listdir(spam1Test):
    f = os.path.join(spam1Test, file)
    if os.path.isfile(f):
        email = open(f, 'r', errors = 'ignore')
        spamtest.append(email.read())
    else:
        print("Testing file could not be found")
        sys.exit()

vec = CountVectorizer(vocabulary=words)
X = vec.transform(spamtest)
spamtestdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

hamclasses = [1]*len(hamtest)
spamclasses = [0]*len(spamtest)

#creating bernoulli dataset
berhamtestdm = hamtestdm.copy()
berhamtestdm[berhamtestdm>0] = 1

berspamtestdm = spamtestdm.copy()
berspamtestdm[berspamtestdm>0] = 1

#applying model to test data
bowTP, bowFN = classification_accuracy(hamtestdm.to_numpy(), hamclasses, bow_weights)
bowTN, bowFP = classification_accuracy(spamtestdm.to_numpy(), spamclasses, bow_weights)

berTP, berFN = classification_accuracy(berhamtestdm.to_numpy(), hamclasses, ber_weights)
berTN, berFP = classification_accuracy(berspamtestdm.to_numpy(), spamclasses, ber_weights)

#print results

bowAccuracy = (bowTP+bowTN)/(bowTP+bowTN+bowFN+bowFP)
bowPrecision = bowTP/(bowTP+bowFP)
bowRecall = bowTP/(bowTP+bowFN)
bowF1 = 2*bowPrecision*bowRecall/(bowPrecision+bowRecall)

berAccuracy = (berTP+berTN)/(berTP+berTN+berFN+berFP)
berPrecision = berTP/(berTP+berFP)
berRecall = berTP/(berTP+berFN)
berF1 = 2*berPrecision*berRecall/(berPrecision+berRecall)

print("Bag of Words = \nAccuracy: " + str(bowAccuracy) + "\nPrecision: " + str(bowPrecision) + "\nRecall: " + str(bowRecall) + "\nF1 Score: " + str(bowF1) + "\n")
print("Bernoulli = \nAccuracy: " + str(berAccuracy) + "\nPrecision: " + str(berPrecision) + "\nRecall: " + str(berRecall) + "\nF1 Score: " + str(berF1))

#testing functions for finding lambda and stuff

#bow_weights = logisticRegression(bowFeaturesTrain, bow_train_classes, initial_weights, learning_rate, lambda_val, max_iterations)
#print(classification_accuracy(bowFeaturesValid, bow_validation_classes, bow_weights))
#ber_weights = logisticRegression(berFeaturesTrain, ber_train_classes, initial_weights, learning_rate, lambda_val, max_iterations)
#print(classification_accuracy(berFeaturesValid, ber_validation_classes, ber_weights))

