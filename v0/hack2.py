# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:33:14 2017

@author: pegasus
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("final.csv")

le = preprocessing.LabelEncoder()
l1 = df["Soil"]
le.fit(l1)
newsoil = le.transform(l1)
df["Soil"]=newsoil

l2 = df["Month"]
le.fit(l2)
df["Month"]=le.transform(l2)

l3 = df["State"]
le.fit(l3)
df["State"]=le.transform(l3)

#df=df.iloc[:,1:]
df = pd.DataFrame(data = df.iloc[:,1:].values, columns=["Soil","Month","State","Rice","Wheat","Cotton","Sugarcane","Tea","Coffee","Cashew","Rubber","Coconut","Oilseed","Ragi","Maize","Groundnut","Millet","Barley"])

feat = pd.DataFrame({"Soil": df["Soil"], "Month" : df["Month"], "State": df["State"]})
#print(feat)
labels = pd.DataFrame(data=df.iloc[:,3:],columns=["Rice","Wheat","Cotton","Sugarcane","Tea",	"Coffee","Cashew","Rubber","Coconut","Oilseed","Ragi","Maize","Groundnut","Millet","Barley"])

features = feat.values
target = labels.values
clf = DecisionTreeClassifier(random_state=42)
#clf = RandomForestClassifier()
print(cross_val_score(clf, features, target, cv=10))

"""
(trainData, testData, trainLabels, testLabels) = train_test_split(feat, labels, test_size=0.25, random_state=42)

totaltrain = pd.concat([trainData, trainLabels], axis=1)
totaltest = pd.concat([testData, testLabels], axis=1)
import tempfile

model_d = tempfile.mkdtemp()

m = tf.estimator.LinearClassifier(
    model_dir=model_d, feature_columns=["Soil","Month","State"],
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.01,
      l1_regularization_strength=1.0,
      l2_regularization_strength=1.0))

m.train(tf.estimator.inputs.pandas_input_fn(x=totaltrain, y=trainLabels,
      batch_size=40,
      num_epochs=None,
      shuffle=True,
      num_threads=5),
    steps=500)
results = m.evaluate(
    tf.estimator.inputs.pandas_input_fn(x=totaltest,
      y=testLabels,
      batch_size=40,
      num_epochs=None,
      shuffle=True,
      num_threads=5),
    steps=None)
    
    
print("model directory = %s" % model_d)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))

"""
