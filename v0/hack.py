# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:41:00 2017

@author: pegasus
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import tempfile
from tensorflow.contrib.learn.python.learn.estimators import random_forest

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

df=df.iloc[:,1:]
#df = pd.DataFrame(data = df.iloc[1:,:].values)

print(df)
df.to_csv('datafile.csv', index=False)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '', 'Base directory for output models.')
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

flags.DEFINE_integer('train_steps', 1000, 'Number of training steps.')
flags.DEFINE_string('batch_size', 1000,
                    'Number of examples in a training batch.')
flags.DEFINE_integer('num_trees', 100, 'Number of trees in the forest.')
flags.DEFINE_integer('max_nodes', 1000, 'Max total nodes in a single tree.')


def build_estimator(model_dir):
  params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
      num_classes=15, num_features=3,
      num_trees=FLAGS.num_trees, max_nodes=FLAGS.max_nodes)
  return random_forest.TensorForestEstimator(params, model_dir=model_dir)


def train_and_eval():
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  print('model directory = %s' % model_dir)

  estimator = build_estimator(model_dir)

  early_stopping_rounds = 100
  check_every_n_steps = 100
  monitor = random_forest.LossMonitor(early_stopping_rounds,
                                      check_every_n_steps)

  feat = df.read_data_sets(FLAGS.data_dir, one_hot=True)

  estimator.fit(x=feat, y=mnist.train.labels,
                batch_size=FLAGS.batch_size, monitors=[monitor])

  results = estimator.evaluate(x=mnist.test.images, y=mnist.test.labels,
                               batch_size=FLAGS.batch_size)
  for key in sorted(results):
    print('%s: %s' % (key, results[key]))


def main(_):
  train_and_eval()


if __name__ == '__main__':
  tf.app.run()