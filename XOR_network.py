import numpy as np
import tensorflow as tf


def buildNeuralNetwork_XOR(matrix, excepted, k, w1_data, w2_data, b1_data, b2_data):
    num_input_neoruns = 2
    num_hidden_neoruns = k
    num_output_neoruns = 1
    nb_hbridge = num_input_neoruns + num_hidden_neoruns
    temperature = 0.001

    # initializing placeholders and variables
    x = tf.placeholder(tf.float32, [None, num_input_neoruns])
    y = tf.placeholder(tf.float32, [None, num_output_neoruns])

    # [num_input_neoruns, num_hidden_neoruns] 
    w1 = tf.Variable(w1_data.reshape([num_input_neoruns, num_hidden_neoruns]), dtype=tf.float32)
    
    #initializing w2: if k = 1, w2 get's bridge (nb_hbridge) and nb_hidden otherwise
    if k == 1:
        w2 = tf.Variable(w2_data.reshape([nb_hbridge, num_output_neoruns]), dtype=tf.float32)

    else: 
        w2 = tf.Variable(w2_data.reshape([num_hidden_neoruns, num_output_neoruns]), dtype=tf.float32)

    #initializing biases and sigmoid
    b1 = tf.Variable(b1_data, dtype=tf.float32)
    b2 = tf.Variable(b2_data, tf.float32)

    z1 = tf.matmul(x, w1) + b1
    hidden_layer1_output = tf.sigmoid(z1 / temperature)
    
    #initializing hlayer1
    if k == 1:
        concated_hidden_layer1_output = tf.concat([hidden_layer1_output, x], 1)

    else:
        concated_hidden_layer1_output = hidden_layer1_output
    
    z2 = tf.matmul(concated_hidden_layer1_output, w2) + b2 

    final_output = tf.sigmoid(z2 / temperature)

    squared_deltas = tf.square(final_output - y)
    loss = tf.reduce_sum(squared_deltas)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess.run([final_output, loss], {x: matrix, y: excepted})


def main():
    # XOR truth table input: 2-dimentional matrix
    x_train = [[0,0],[0,1],[1,0],[1,1]]
    expected = [[0], [1], [1], [0]]

    # num_hidden_neoruns
    print("\nk (num_hidden_neoruns) = 1:\n")
    k = 1  
    w1_data = np.array([1, 1])
    w2_data = np.array([-2, 1, 1])
    b1_data = [-1.5] 
    b2_data = [-.5]
    results = buildNeuralNetwork_XOR(x_train, expected, k, w1_data, w2_data, b1_data, b2_data)
    print("\nactual = " + str(results[0]) + "\nexpected = " + str(expected) + "\nloss = " + str(results[1]))

    # num_hidden_neoruns
    print("\nk (num_hidden_neoruns) = 2:\n")
    k = 2  
    w1_data = np.array([1, -1, 1, -1])
    w2_data = np.array([1, 1])
    b1_data = [-.5, 1.5]
    b2_data = [-1.5]
    results = buildNeuralNetwork_XOR(x_train, expected, k, w1_data, w2_data, b1_data, b2_data)
    print("\nactual = " + str(results[0]) + "\nexpected = " + str(expected) + "\nloss = " + str(results[1]))

    # num_hidden_neoruns
    print("\nk (num_hidden_neoruns) = 4:\n")
    k = 4  
    w1_data = np.array([-1., -1., 1., 1., -1., 1., -1., 1.])
    w2_data = np.array([-1, 1, 1, -1])
    b1_data = [-.5, -.5, -.5, -2.5]
    b2_data = [-.5]
    results = buildNeuralNetwork_XOR(x_train, expected, k, w1_data, w2_data, b1_data, b2_data)
    print("actual = " + str(results[0]) + "\nexpected = " + str(expected) + "\nloss = " + str(results[1]))


if __name__ == '__main__':
    main()