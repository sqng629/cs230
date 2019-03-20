import csv
import random

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, BatchNormalization, Activation

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

    inputs = np.genfromtxt(csv_path, delimiter=',', usecols=x_cols, skip_header=1, missing_values="?", filling_values=-1, dtype='float32')

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


    #data augmentation
    
    for i in range(2):
        added_inputs = np.zeros(train_inputs.shape)
    
        for i in range(train_inputs.shape[0]):
            noise = np.random.uniform(low=-4, high=4, size=(1,train_inputs.shape[1]))
            added_inputs[i,:] = np.add(train_inputs[i,:], noise)
    
        train_inputs = np.concatenate((train_inputs, added_inputs), axis=0)
        train_labels = np.concatenate((train_labels, train_labels), axis=0)

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

    epochs = 300
    display_step = 100
    batch_size = 50
    
    x = tf.placeholder(tf.float32, [num_features, None])
    y = tf.placeholder(tf.float32, [1, None])
    
    predictions = model(x, keep_prob, num_features)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
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
                                keep_prob: .8
                                })
                avg_cost += c / num_batches
            if epoch % display_step == 0:
                print("epoch number=", '%d' % (epoch+1), "average cost=", "{:.6f}".format(avg_cost))

        
        #calculate train and test error
        correct_prediction = tf.equal(tf.round(tf.sigmoid(predictions)), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test Accuracy:", accuracy.eval({x: test_inputs, y: test_labels, keep_prob: 1.0}))
        print("Train Accuracy:", accuracy.eval({x: train_inputs, y: train_labels, keep_prob: 1.0}))

def runKerasCNNModel(train_inputs, train_labels, test_inputs, test_labels):
    
    num_inputs = train_inputs.shape[1]
    num_features = train_inputs.shape[0]
    
    train_inputs = train_inputs.T
    test_inputs = test_inputs.T
    train_labels = train_labels.T
    test_labels = test_labels.T
    
    train_inputs = np.expand_dims(train_inputs, axis=2)
    test_inputs = np.expand_dims(test_inputs, axis=2)
    
    model = Sequential()
    
    model.add(Conv1D(kernel_size=(3), filters=5, padding='same', input_shape=(num_features,1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv1D(kernel_size=(3), filters=6, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size = (4), strides=(2)))
    #model.add(Conv1D(kernel_size=(3), filters=8, padding='same'))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    #model.add(Conv1D(kernel_size=(3), filters=6, padding='same'))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    #model.add(MaxPooling1D(pool_size = (4), strides=(2)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=20)


def main():
    inputs, labels = load_csv('features.csv')

    train_inputs, train_labels, test_inputs, test_labels = generateTrainingAndTestingDataSet(inputs, labels)
    #train tensorflow network
    train_inputs = train_inputs.T
    train_labels = np.reshape(train_labels, [1, len(train_labels)])
    test_inputs = test_inputs.T
    test_labels = np.reshape(test_labels, [1, len(test_labels)])
    #runTensorflowModel(train_inputs, train_labels, test_inputs, test_labels)
    runKerasCNNModel(train_inputs, train_labels, test_inputs, test_labels)

if __name__ == "__main__":
    main()
