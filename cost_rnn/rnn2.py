"""
This file basically replicates a basic distance algorithm and sends the output
to the correct interface.
"""

from __future__ import print_function

import protos.model_pb2 as model_pb2
import google.protobuf.text_format as text_format
import pkgutil
import tensorflow as tf
import numpy as np
import random
import sys

class ProtoSequenceData(object):
    def __init__(self, examples):
        self.index = 0
        self.data = []
        self.labels = []
        self.seqlen = len(examples[0].feature_lists.feature_list["interface_ids"].feature[0].float_list.value)
        for example in examples:
            data = []
            label = None
            for i in range(self.seqlen):
                point = []
                for field in ("interface_ids", "size", "value"):
                    point.append(example.feature_lists.feature_list[field].feature[0].float_list.value[i])
                label = example.feature_lists.feature_list["delivered_value"].feature[0].float_list.value[i]
                data.append(point)
            self.data.append(data)
            self.labels.append([label])

    def next(self, batch_size):
        if self.index == len(self.data):
            self.index = 0
        batch_data = self.data[self.index: self.index+batch_size]
        batch_labels = self.labels[self.index: self.index+batch_size]
        batch_seqlen =  [self.seqlen for i in range(batch_size)]
        self.index += batch_size
        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.001
training_iters = 1000 * 200
batch_size =200
display_step = 10

# Network Parameters
seq_max_len = 100
n_hidden = 64 # hidden layer num of features
n_classes = 1 # linear sequence or not

f = open(sys.argv[1], "rb")
model = model_pb2.Model()
text = f.read()
text_format.Merge(text.decode(), model)
f.close()

trainset = ProtoSequenceData(model.examples[batch_size:len(model.examples)])
testset = ProtoSequenceData(model.examples[0:batch_size])
print('data loaded')

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, seq_max_len, 3], name="x")
y = tf.placeholder(tf.float32, [batch_size, 1], name="y")
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [batch_size])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)

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
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
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
cost = tf.reduce_mean(tf.square(pred- y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
accuracy = tf.reduce_mean(tf.square(pred- y))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    print('Starting')
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
            test_data, test_label, test_seqlen = testset.next(batch_size)
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                              seqlen: test_seqlen}))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data, test_label, test_seqlen = testset.next(batch_size)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
