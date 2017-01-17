"""
This file basically replicates a basic distance algorithm and sends the output
to the correct interface.
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random


class ToySequenceData(object):
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000, max_interfaces=10):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.batch_id = 0
        for i in range(n_samples):
            distance = []
            reachable = []

            # Random sequence length
            num_interfaces = random.randint(min_seq_len, max_interfaces)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(num_interfaces)

            max_distance = max_value + 1
            destination = 0
            data = []
            for i in  range(num_interfaces):
                d = random.randint(0, max_value)
                r = 1 if random.random() < 0.5 else 0
                distance.append(d)
                reachable.append(r)
                if (max_distance > d and r >0):
                    destination = i +1
                    max_distance = d
                point = [0 if i !=j else 1 for j in range(max_interfaces)]
                point.append(r)
                point.append(d)
                data.append(point)
            labels = [0 if i != destination else 1 for i in range(max_interfaces)]

            for i in range(max_seq_len - num_interfaces ):
                data.append([0 for j in range(max_interfaces+2)])

            self.data.append(data)
            self.labels.append(labels)

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))]
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        print(batch_seglen)
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 1000000000
batch_size = 1000
display_step = 100

# Network Parameters
seq_max_len = 20 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 16 # linear sequence or not

trainset = ToySequenceData(n_samples=100000, max_seq_len=seq_max_len, max_interfaces = n_classes)
testset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len, max_interfaces = n_classes)

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, seq_max_len, n_classes+2])
y = tf.placeholder(tf.float32, [batch_size, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [batch_size])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_classes+2])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(axis=0, num_or_size_splits=seq_max_len, value=x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            # Calculate accuracy
            test_data = testset.data
            test_label = testset.labels
            test_seqlen = testset.seqlen
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                              seqlen: test_seqlen}))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
