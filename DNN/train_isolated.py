from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
                                                       filename="output.csv",
                                                       features_dtype=np.int,
                                                       target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
                                                   filename="test.csv",
                                                   features_dtype=np.int,
                                                   target_dtype=np.int)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/routing_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

new_samples = np.array(
        [[1, 10], [0, 1]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
