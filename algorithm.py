import csv
import random

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

def notTargetVariable(headerName):
    if(headerName != "TextID" and headerName != "URL" and headerName != "Label"):
        return 1
    else:
        return 0

def notInList(element, list):
    if not element in list:
        return True
    else:
        return False


def load_csv(csv_path, label_col='Label'):
    
    # Load headers
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if notTargetVariable(headers[i])]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]

    inputs = np.genfromtxt(csv_path, delimiter=',', usecols=x_cols, skip_header=1, missing_values="?", filling_values=-1, dtype='int32')
    
    labels = np.genfromtxt(csv_path, delimiter=',', usecols=l_cols, skip_header=1, dtype='str')
    labels = [int(x=='objective') for x in labels]
    labels = np.asarray(labels)

    return inputs, labels



def generateTrainingAndTestingDataSet(inputs, labels):
    #randomly select inputs to go into test set
    num_inputs = inputs.shape[0]
    all_indexes = range(num_inputs)
    test_indexes = random.sample(all_indexes, np.int(num_inputs * .05))
    train_indexes = [i for i in range(num_inputs) if notInList(i, test_indexes)]
    
    train_inputs = inputs[train_indexes,:]
    test_inputs = inputs[test_indexes,:]

    train_labels = labels[train_indexes]
    test_labels = labels[test_indexes]

    return (train_inputs, train_labels, test_inputs, test_labels)

def model(x, keep_prob, num_features):
    
    l1_hidden_units = 300
    l2_hidden_units = 200
    
    weights = {
        'w1': tf.get_variable("w1", shape=[l1_hidden_units, num_features], initializer=tf.contrib.layers.xavier_initializer()),
        'w2': tf.get_variable("w2", shape=[l2_hidden_units, l1_hidden_units], initializer=tf.contrib.layers.xavier_initializer()),
        'w3': tf.get_variable("w3", shape=[1, l2_hidden_units], initializer=tf.contrib.layers.xavier_initializer())
    }
    
    biases = {
        'b1': tf.get_variable("b1", shape=[l1_hidden_units, 1], initializer=tf.zeros_initializer()),
        'b2': tf.get_variable("b2", shape=[l2_hidden_units, 1], initializer=tf.zeros_initializer()),
        'b3': tf.get_variable("b3", shape=[1, 1], initializer=tf.zeros_initializer())
    }

    #layer 1
    layer = tf.matmul(weights['w1'], x)
    layer = tf.add(layer, biases['b1'])
    layer = tf.nn.sigmoid(layer)
    layer = tf.nn.dropout(layer, keep_prob)
    #layer 2
    layer = tf.matmul(weights['w2'], layer)
    layer = tf.add(layer, biases['b2'])
    layer = tf.nn.sigmoid(layer)
    layer = tf.nn.dropout(layer, keep_prob)
    out_layer = tf.add(tf.matmul(weights['w3'], layer), biases['b3'])
    return out_layer

def runTensorflowModel(train_inputs, train_labels, test_inputs, test_labels):
    
    num_inputs = train_inputs.shape[1]
    num_features = train_inputs.shape[0]
    
    keep_prob = tf.placeholder("float")

    epochs = 10000
    display_step = 1000
    batch_size = 50
    
    x = tf.placeholder(tf.float32, [num_features, None])
    y = tf.placeholder(tf.float32, [1, None])
    
    predictions = model(x, keep_prob, num_features)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            avg_cost = 0.0
            num_batches = int(num_inputs / batch_size)
            
            x_batches = np.array_split(train_inputs, num_batches, axis=1)
            y_batches = np.array_split(train_labels, num_batches, axis=1)
            for i in range(num_batches):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 0.9
                                })
                avg_cost += c / num_batches
            if epoch % display_step == 0:
                print("epoch number=", '%d' % (epoch+1), "average cost=", "{:.6f}".format(avg_cost))

        
        #calculate train and test error
        correct_prediction = tf.equal(tf.round(tf.sigmoid(predictions)), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test Accuracy:", accuracy.eval({x: test_inputs, y: test_labels, keep_prob: 1.0}))
        print("Train Accuracy:", accuracy.eval({x: train_inputs, y: train_labels, keep_prob: 1.0}))

def main():
    inputs, labels = load_csv('features.csv')

    train_inputs, train_labels, test_inputs, test_labels = generateTrainingAndTestingDataSet(inputs, labels)
    #train tensorflow network
    train_inputs = train_inputs.T
    train_labels = np.reshape(train_labels, [1, len(train_labels)])
    test_inputs = test_inputs.T
    test_labels = np.reshape(test_labels, [1, len(test_labels)])
    runTensorflowModel(train_inputs, train_labels, test_inputs, test_labels)


if __name__ == "__main__":
    main()
