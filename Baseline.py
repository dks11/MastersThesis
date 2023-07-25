import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier


from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2

import math

from datetime import datetime
start_time = datetime.now()

dfTrain = pd.read_csv("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Features\\enron1\\Train.csv")
dfTrain.dropna(inplace=True)
dfTrain["spam/ham"] = pd.to_numeric(dfTrain["spam/ham"])
#dfTrain = dfTrain.sample(n = 300).reset_index(drop=True)


ham = len(dfTrain[dfTrain['spam/ham'] == 0])
spam = len(dfTrain[dfTrain['spam/ham'] == 1])
totalTrain = ham + spam
print(f"Train Total = {totalTrain}")
hamratio = ham / totalTrain
spamratio = spam / totalTrain
print(f"Ham:{ham}")
print(f"Spam:{spam}")
print(f"Ham Ratio:{hamratio}")
print(f"Spam Ratio:{spamratio}\n")

X_Train = dfTrain.drop('spam/ham', axis=1)
Y_Train = dfTrain['spam/ham']

dfTest = pd.read_csv("C:\\Users\\karen\\Desktop\\Thesis\\Datasets\\Final TrainTest Data\\Features\\enron1\\Test.csv")
dfTest.dropna(inplace=True)
dfTest["spam/ham"] = pd.to_numeric(dfTest["spam/ham"])
dfTest = dfTest.sample(frac=1).reset_index(drop=True)

ham = len(dfTest[dfTest['spam/ham'] == 0])
spam = len(dfTest[dfTest['spam/ham'] == 1])
total = ham + spam
print(f"Test Total = {total}")
hamratio = ham / total
spamratio = spam / total
print(f"Ham:{ham}")
print(f"Spam:{spam}")
print(f"Ham Ratio:{hamratio}")
print(f"Spam Ratio:{spamratio}\n")

X_Test = dfTest.drop('spam/ham', axis=1)
Y_Test = dfTest['spam/ham']


print("Beginning Training")
lr = LogisticRegression(max_iter=100000, solver = "liblinear")
lrCLF = lr.fit(X_Train, Y_Train)

#tr = tree.DecisionTreeClassifier()
#treeCLF = tr.fit(X_Train, Y_Train)

rf = RandomForestClassifier(n_estimators=100, random_state = 42)
rfCLF = rf.fit(X_Train, Y_Train)

#bnb = BernoulliNB()
#bnbCLF = bnb.fit(X_Train, Y_Train)

SVMModel = svm.SVC(random_state = 42)
svmCLF = SVMModel.fit(X_Train, Y_Train)

#knn = KNeighborsClassifier(n_neighbors = math.floor(math.sqrt(totalTrain)))
#knnCLF = knn.fit(X_Train, Y_Train)

#ada = AdaBoostClassifier()
#adaCLF = ada.fit(X_Train, Y_Train)

#ensemble = VotingClassifier(estimators=[('lr', lrCLF), ('rf', rfCLF), ('bnb', bnbCLF), ('SVMModel', svmCLF), ('knn',knnCLF), ('tr',treeCLF), ('ada',adaCLF)])
#eCLF = ensemble.fit(X_Train, Y_Train)


print("Beginning Testing\n")
resultslr = lrCLF.predict(X_Test)
#resultstree = treeCLF.predict(X_Test)
resultsrf = rfCLF.predict(X_Test)
#resultsbnb = bnbCLF.predict(X_Test)
resultssvm = svmCLF.predict(X_Test)
#resultsensemble = eCLF.predict(X_Test)
#resultsknn = knnCLF.predict(X_Test)
#resultsada = adaCLF.predict(X_Test)

print("LogisticRegression")
print("Accuracy: ",accuracy_score(Y_Test, resultslr))

print("Precision Score 1: ",precision_score(Y_Test, resultslr, pos_label = 1))
print("Precision Score 0: ",precision_score(Y_Test, resultslr, pos_label = 0))

print("Recall Score 1: ",recall_score(Y_Test, resultslr, pos_label = 1))
print("Recall Score 0: ",recall_score(Y_Test, resultslr, pos_label = 0))


print("F1 Score: ",f1_score(Y_Test, resultslr, pos_label = 1))
print("F1 Score: ",f1_score(Y_Test, resultslr, pos_label = 0))

print(confusion_matrix(Y_Test, resultslr))

print("\n")

#print("Decision Tree")
#print("Accuracy: ",accuracy_score(Y_Test, resultstree))

#print("Precision Score 1: ",precision_score(Y_Test, resultstree, pos_label = 1))
#print("Precision Score 0: ",precision_score(Y_Test, resultstree, pos_label = 0))

#print("Recall Score 1: ",recall_score(Y_Test, resultstree, pos_label = 1))
#print("Recall Score 0: ",recall_score(Y_Test, resultstree, pos_label = 0))


#print("F1 Score: ",f1_score(Y_Test, resultstree, pos_label = 1))
#print("F1 Score: ",f1_score(Y_Test, resultstree, pos_label = 0))

#print(confusion_matrix(Y_Test, resultstree))

#print("\n")

print("RandomForestClassifier")
print("Accuracy: ",accuracy_score(Y_Test, resultsrf))

print("Precision Score 1: ",precision_score(Y_Test, resultsrf, pos_label = 1))
print("Precision Score 0: ",precision_score(Y_Test, resultsrf, pos_label = 0))

print("Recall Score 1: ",recall_score(Y_Test, resultsrf, pos_label = 1))
print("Recall Score 0: ",recall_score(Y_Test, resultsrf, pos_label = 0))

print("F1 Score 1: ",f1_score(Y_Test, resultsrf, pos_label = 1))
print("F1 Score 0: ",f1_score(Y_Test, resultsrf, pos_label = 0))

print(confusion_matrix(Y_Test, resultsrf))

print("\n")

#print("BernoulliNB")
#print("Accuracy: ",accuracy_score(Y_Test, resultsbnb))

#print("Precision Score 1: ",precision_score(Y_Test, resultsbnb, pos_label = 1))
#print("Precision Score 0: ",precision_score(Y_Test, resultsbnb, pos_label = 0))

#print("Recall Score 1: ",recall_score(Y_Test, resultsbnb, pos_label = 1))
#print("Recall Score 0: ",recall_score(Y_Test, resultsbnb, pos_label = 0))

#print("F1 Score 1: ",f1_score(Y_Test, resultsbnb, pos_label = 1))
#print("F1 Score 0: ",f1_score(Y_Test, resultsbnb, pos_label = 0))

#print(confusion_matrix(Y_Test, resultsbnb))

#print("\n")

print("SVM")
print("Accuracy: ",accuracy_score(Y_Test, resultssvm))

print("Precision Score 1: ",precision_score(Y_Test, resultssvm, pos_label = 1))
print("Precision Score 0: ",precision_score(Y_Test, resultssvm, pos_label = 0))

print("Recall Score 1: ",recall_score(Y_Test, resultssvm, pos_label = 1))
print("Recall Score 0: ",recall_score(Y_Test, resultssvm, pos_label = 0))

print("F1 Score 1: ",f1_score(Y_Test, resultssvm, pos_label = 1))
print("F1 Score 0: ",f1_score(Y_Test, resultssvm, pos_label = 0))

print(confusion_matrix(Y_Test, resultssvm))

print("\n")

#print("KNN")
#print("Accuracy: ",accuracy_score(Y_Test, resultsknn))

#print("Precision Score 1: ",precision_score(Y_Test, resultsknn, pos_label = 1))
#print("Precision Score 0: ",precision_score(Y_Test, resultsknn, pos_label = 0))

#print("Recall Score 1: ",recall_score(Y_Test, resultsknn, pos_label = 1))
#print("Recall Score 0: ",recall_score(Y_Test, resultsknn, pos_label = 0))

#print("F1 Score 1: ",f1_score(Y_Test, resultsknn, pos_label = 1))
#print("F1 Score 0: ",f1_score(Y_Test, resultsknn, pos_label = 0))

#print(confusion_matrix(Y_Test, resultsknn))

#print("\n")

#print("AdaBoost")
#print("Accuracy: ",accuracy_score(Y_Test, resultsada))

#print("Precision Score 1: ",precision_score(Y_Test, resultsada, pos_label = 1))
#print("Precision Score 0: ",precision_score(Y_Test, resultsada, pos_label = 0))

#print("Recall Score 1: ",recall_score(Y_Test, resultsada, pos_label = 1))
#print("Recall Score 0: ",recall_score(Y_Test, resultsada, pos_label = 0))

#print("F1 Score 1: ",f1_score(Y_Test, resultsada, pos_label = 1))
#print("F1 Score 0: ",f1_score(Y_Test, resultsada, pos_label = 0))

#print(confusion_matrix(Y_Test, resultsada))

#print("\n")

#print("Ensemble")
#print("Accuracy: ",accuracy_score(Y_Test, resultsensemble))

#print("Precision Score 1: ",precision_score(Y_Test, resultsensemble, pos_label = 1))
#print("Precision Score 0: ",precision_score(Y_Test, resultsensemble, pos_label = 0))

#print("Recall Score 1: ",recall_score(Y_Test, resultsensemble, pos_label = 1))
#print("Recall Score 0: ",recall_score(Y_Test, resultsensemble, pos_label = 0))


#print("F1 Score 1: ",f1_score(Y_Test, resultsensemble, pos_label = 1))
#print("F1 Score 0: ",f1_score(Y_Test, resultsensemble, pos_label = 0))

#print(confusion_matrix(Y_Test, resultsensemble))



end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))