import tensorflow as tf
import numpy as np
import sys

from tensorflow.python.ops import rnn_cell, rnn


reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["output.tfrecord"],
						num_epochs=None)
_, serialized_example = reader.read(filename_queue)
context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
    sequence_features={
        "interface_identifier" : tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "reachable" : tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "distance" : tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "active_output" : tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "selected_interface" : tf.FixedLenSequenceFeature([], dtype=tf.float32),
        })

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 7
display_step = 1

# Network Parameters
n_input = 4
n_steps = 17
n_hidden = 128 # hidden layer num of features

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, 1]))
}
biases = {
    'out': tf.Variable(tf.random_normal([1]))
}

print('raw', sequence_parsed)
batch = tf.train.batch(tensors=sequence_parsed,
        batch_size= batch_size,
        dynamic_pad=True)
print('batch', batch)
input_batch =  tf.pack([
    batch['interface_identifier'],
    batch['reachable'],
    batch['distance'],
    batch['active_output'],
    ])
output_batch = batch['selected_interface']
print('Input batch dimension', input_batch)
print('output batch dimension', output_batch)
input_batch = tf.transpose(input_batch, perm=[2, 0, 1])
output_batch = tf.transpose(output_batch, perm=[1, 0])
print('Input batch dimension', input_batch)
print('output batch dimension', output_batch)
input_batch = tf.squeeze(input_batch, [2])
output_batch = tf.squeeze(output_batch, [1])
print('Input batch dimension', input_batch)
print('output batch dimension', output_batch)


# Define a lstm cell with tensorflow
lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

# Get lstm cell output
outputs, states = rnn.dynamic_rnn(lstm_cell, input_batch, dtype=tf.float32)

# Linear activation, using rnn inner loop last output
lin_act = tf.matmul(outputs[-1], weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lin_act, output_batch[-1]))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(lin_act,1), tf.argmax(output_batch,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.train.start_queue_runners()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        print ('starting')
        sess.run(optimizer, feed_dict=None)
        print ('one run')
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict=None)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict=None)
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
