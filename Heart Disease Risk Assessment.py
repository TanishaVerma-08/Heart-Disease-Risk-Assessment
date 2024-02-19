import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sklearn as sk
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pickle

df = pd.read_csv("D:\Practice ML\CodeClause\Heart Disease Risk Management\heart.csv")
df.head(10)
df.isna().sum()
df.shape
df.info()
df.describe().T

"""There are no missing values"""

df['target'].unique()

df['target'].value_counts(normalize = True)

"""1 indicates Risk

0 indicates No Risk
"""
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg','exang', 'slope', 'ca', 'thal'])

y = df['target']
X = df.drop('target',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

LR_classifier = LogisticRegression(random_state=0)
clf = svm.SVC()

sgd=SGDClassifier()
forest=RandomForestClassifier(n_estimators=20, random_state=12,max_depth=6)

treee = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
LR_classifier.fit(X_train, y_train)

clf.fit(X_train, y_train)
sgd.fit(X_train, y_train)

treee.fit(X_train, y_train)
forest.fit(X_train, y_train)

#traing accuracy
y_pred=LR_classifier.predict(X_train)
y_predsvm=clf.predict(X_train)
y_predsgd=sgd.predict(X_train)
y_predtree=treee.predict(X_train)
y_predforest=forest.predict(X_train)

print(accuracy_score(y_train, y_pred))
print(accuracy_score(y_train, y_predsvm))
print(accuracy_score(y_train, y_predsgd))
print(accuracy_score(y_train, y_predtree))
print(accuracy_score(y_train, y_predforest))

#test accuracy
y_pred=LR_classifier.predict(X_test)
y_predsvm=clf.predict(X_test)
y_predsgd=sgd.predict(X_test)
y_predtree=treee.predict(X_test)
y_predforest=forest.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_predsvm))
print(accuracy_score(y_test, y_predsgd))
print(accuracy_score(y_test, y_predtree))
print(accuracy_score(y_test, y_predforest))

pickle.dump(forest, open('Random_forest_model.pkl','wb'))