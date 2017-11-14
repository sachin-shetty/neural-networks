import scipy
from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import glob
import sys,os
import pickle

BASE_DATA_DIR="/home/egnyte/work/research/neuralnetworks/arnav_classifier_nn"

# GRADED FUNCTION: random_mini_batches

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name = "Y")

    return X, Y

def initialize_parameters():

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [50,49152], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [100,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [100,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [100,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [100,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [100,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [100,1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [1,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b5 = tf.get_variable("b5", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                }

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                            # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                         # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                       # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)                                       # Z3 = np.dot(W3,Z2) + b3
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5)                                       # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z5

def compute_cost(Z5, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z5)
    labels = tf.transpose(Y)

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    ### END CODE HERE ###

    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 200, minibatch_size = 64, print_cost = True):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z5 = forward_propagation(X, parameters)
    cost = compute_cost(Z5, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
                ### END CODE HERE ###

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")


        # Calculate the correct predictions
        A5 = tf.sigmoid(Z5)
        YHat = A5 > 0.5
        Y_prediction_train = sess.run(YHat, feed_dict = {X: X_train} )
        Y_prediction_test = sess.run(YHat, feed_dict = {X: X_test} )
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        return parameters



def load_image_dir(name):
    #image_list = np.empty([12288, 0])
    image_list = np.empty([49152, 0])
    for filename in glob.glob('%s/*.jpg' % name): #assuming gif
        image = np.array(ndimage.imread(filename, flatten=False))
        my_image = scipy.misc.imresize(image, size=(128,128)).reshape((1, 128*128*3)).T
        my_image = my_image.astype(float)/255
        image_list = np.hstack((image_list, my_image))
        print "Loaded Image: %s" % filename
    return image_list

def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    W5 = tf.convert_to_tensor(parameters["W5"])
    b5 = tf.convert_to_tensor(parameters["b5"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4,
              "W5": W5,
              "b5": b5,
            }

    x = tf.placeholder("float", [49152, 1])

    Z5 = forward_propagation_for_predict(x, params)
    p = tf.sigmoid(Z5)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    print ("Prediction: %s" % prediction)
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                            # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                         # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                       # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)                                       # Z3 = np.dot(W3,Z2) + b3
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5)                                       # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z5



def load_dataset():
    arnav_train = load_image_dir("%s/%s" % (BASE_DATA_DIR, "train/arnav"))
    Y_train = np.ones([1, arnav_train.shape[1]])
    sanu_train = load_image_dir("%s/%s" % (BASE_DATA_DIR, "train/sanu"))
    X_train = np.hstack((arnav_train, sanu_train))
    Y_train = np.hstack((Y_train, np.zeros([1,sanu_train.shape[1]])))

    arnav_test = load_image_dir("%s/%s" % (BASE_DATA_DIR, "test/arnav"))
    Y_test = np.ones([1, arnav_test.shape[1]])
    sanu_test = load_image_dir("%s/%s" % (BASE_DATA_DIR, "test/sanu"))
    X_test = np.hstack((arnav_test, sanu_test))
    Y_test = np.hstack((Y_test, np.zeros([1, sanu_test.shape[1]])))


    print arnav_train.shape
    print sanu_train.shape
    print Y_train.shape
    #print Y_train

    print arnav_test.shape
    print sanu_test.shape
    print Y_test.shape
    #print Y_test

    return (X_train, Y_train, X_test, Y_test, 2)


def predict_for_testing(fname, parameters):
    # We preprocess your image to fit your algorithm.
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(128,128)).reshape((1, 128*128*3)).T
    my_image = my_image.astype(float)/255

    my_image_prediction = predict(my_image, parameters)

    if (np.squeeze(my_image_prediction) >= 0.5):
        my_image_prediction = "Arnav"
    else:
        my_image_prediction = "Sanu"

    print("Your algorithm predicts: y = " + my_image_prediction + " for " + fname)
    plt.imshow(image)
    plt.show()

def test_for_dir(base_dir):
    for root, dir, files in os.walk(base_dir):
        for item in files:
            fname = "%s/%s" % (base_dir, item)
            
            predict_for_testing(fname, parameters)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == "reuse":
        pickle_in = open("parameters.pickle","rb")
        parameters = pickle.load(pickle_in)    
    else:
        (X_train, Y_train, X_test, Y_test, classes) = load_dataset()
        parameters = model(X_train, Y_train, X_test, Y_test)
        pickle_out = open("parameters.pickle","wb")
        pickle.dump(parameters, pickle_out)
        pickle_out.close()

    #test_for_dir(BASE_DATA_DIR + "/real_world_test")
    test_for_dir(BASE_DATA_DIR + "/test/sanu")


"""
fname = "train/arnav/IMG_20170520_001651.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

plt.imshow(scipy.misc.imresize(image, size=(128,128)))
plt.show()
print my_image.shape
"""
