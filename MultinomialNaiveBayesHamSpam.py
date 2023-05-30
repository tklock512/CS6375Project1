import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import math

#start of main

#getting data

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

#creating Bag of Words model for hams

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

vec = CountVectorizer()
X = vec.fit_transform(hams)
hamsdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

#creating Bag of Words model for spams
for file in os.listdir(spam1Train):
    f = os.path.join(spam1Train, file)
    if os.path.isfile(f):
        email = open(f, 'r', errors = 'ignore')
        spams.append(email.read())
    else:
        print("Training file could not be found")
        sys.exit()

vec = CountVectorizer()
X = vec.fit_transform(spams)
spamsdm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())


#Making Multinomial Naive Bayes model

features = {}
features['ham'] = {}
features['spam'] = {}

hamCount = hamsdm.shape[0]
spamCount = spamsdm.shape[0]
totalCount = hamCount + spamCount

logPriorHam = math.log(hamCount/totalCount)
logPriorSpam = math.log(spamCount/totalCount)

#calculate probabilities
wordHamCounts = hamsdm.sum()
for word, count in wordHamCounts.items():
    features['ham'][word] = math.log((count + 1)/(hamCount+totalCount))

wordSpamCounts = spamsdm.sum()
for word, count in wordSpamCounts.items():
    features['spam'][word] = math.log((count + 1)/(spamCount+totalCount))

#Calculate confusion matrix/results

trueNegative = 0
falsePositive = 0
falseNegative = 0
truePositive = 0

#iterate through test hams

for file in os.listdir(ham1Test):
    fileLoc = os.path.join(ham1Test, file)
    #best way I could find to handle special characters that require different encodings
    with open(fileLoc, errors = 'ignore') as f:
        txt=[word for line in f for word in line.split()]

    ham_score = logPriorHam
    spam_score = logPriorSpam

    for f in features:
        if f == 'ham':
            for word in txt:
                if word in features['ham']:
                    ham_score -= features['ham'][word]
        if f =='spam':
            for word in txt:
                if word in features['spam']:
                    spam_score -= features['spam'][word]

    if ham_score >= spam_score:
        #classify as ham
        truePositive += 1
    elif spam_score > ham_score:
        #classify as spam
        falseNegative += 1

#iterate through test spams

for file in os.listdir(spam1Test):
    fileLoc = os.path.join(spam1Test, file)

    with open(fileLoc, errors = 'ignore') as f:
        txt=[word for line in f for word in line.split()]

    ham_score = logPriorHam
    spam_score = logPriorSpam

    for f in features:
        if f == 'ham':
            for word in txt:
                if word in features['ham']:
                    ham_score -= features['ham'][word]
        if f =='spam':
            for word in txt:
                if word in features['spam']:
                    spam_score -= features['spam'][word]

    if ham_score >= spam_score:
        #classify as ham
        falsePositive += 1
    elif spam_score > ham_score:
        #classify as spam
        trueNegative += 1

#print results

print("True Positive: " + str(truePositive) + "\nTrue Negative: " + str(trueNegative) + "\nFalse Positive: " + str(falsePositive) + "\nFalse Negative: " + str(falseNegative))