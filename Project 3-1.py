# Worked together with B081705

# Import for network 1 and 2
import random
import math

# Import for network 3
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import tensorflow as tf


# Network 1 and 2

# Make sigmoid function taking n as input
def sigmoid(n):
    return 1 / (1 + pow(math.e, -n))


# Link object connecting nodes
class Link(object):

    def __init__(self, input_node, output_node, weight):
        self.input_node = input_node  # node providing input
        self.output_node = output_node  # node receiving input
        self.weight = weight  # strength of link between nodes
        self.delta_weight = 0.0  # change in weight for back_prop


# Node object for nodes in layers
class Node(object):

    def __init__(self, node_type):
        self.node_type = node_type  # type of layer node is in ('input', 'hidden', or 'output')
        self.act = 0.0
        self.input = 0.0
        self.error = 0.0
        self.delta = 0.0
        self.weights_above = []
        self.weights_below = []

    def act_fun(self):
        # activation function is sigmoid function of input
        self.act = sigmoid(self.input)

    def input_fun(self):
        # input is sum of activations times weights of nodes in layer below
        self.input = 0.0
        for weight in self.weights_below:
            self.input += weight.weight * weight.input_node.act

    def hidden_node_error(self):

        # function calculating delta for nodes in hidden layer (sum of delta times weights of nodes above)
        if self.node_type == 'hidden':
            self.error = 0.0

            for weight in self.weights_above:
                self.error += weight.weight * weight.output_node.delta


# Network object
class Network(object):

    def __init__(self, num_layers, nodes_hidden, nodes_inp, nodes_out):
        self.num_layers = num_layers  # number of total layers in network
        self.nodes_hidden = nodes_hidden  # number of nodes for all hidden layers
        self.nodes_inp = nodes_inp  # nodes in the input layer
        self.nodes_out = nodes_out  # nodes in the output layer
        self.layer_list = []  # list of layers in the network
        self.weight_list = []  # list of all weights in network

    def create_layers(self):

        # function that creates all layers in the network
        # loop creates 1 layer, iterations = number of layers in network (self.num_layers)
        for i in range(self.num_layers):

            # create empty list that is layer
            layer_x = []

            # depending on how many layers have been created and how many we want in total, different layer is created
            # create as many nodes as specified in init for each layer, add node to layer then add layer to layer list
            if i == 0:
                for j in range(self.nodes_inp):
                    node_x = Node('input')
                    layer_x.append(node_x)
            elif i == self.num_layers - 1:
                for j in range(self.nodes_out):
                    node_x = Node('output')
                    layer_x.append(node_x)
            else:
                for j in range(self.nodes_hidden):
                    node_x = Node('hidden')
                    layer_x.append(node_x)

            self.layer_list.append(layer_x)

    def create_weights(self):

        # create links between all nodes in layer i and layer i + 1
        # loop is 1 less than self.num_layers as we have 1 less layer of links than layer of nodes
        for i in range(self.num_layers - 1):

            for node_i in self.layer_list[i]:
                for node_j in self.layer_list[i + 1]:
                    weight_x = Link(node_i, node_j, random.uniform(-1.0, 1.0))  # weight is random number -1.0 to 1.0
                    self.weight_list.append(weight_x)
                    node_i.weights_above.append(weight_x)
                    node_j.weights_below.append(weight_x)

    def init_input(self, pattern):

        # take pattern and set each input node's activation to corresponding value in pattern
        for i, node in enumerate(self.layer_list[0]):

            node.act = pattern[i]

    def forward_prop(self):

        # forward propagate input by updating input then activation for nodes in layers above input layer, in turn
        for i in range(1, len(self.layer_list)):

            for node in self.layer_list[i]:
                node.input_fun()
            for node in self.layer_list[i]:
                node.act_fun()

    def back_prop(self, pattern_list, desired_list, mse_min, max_iterations, eta, momentum):
        # back_prop function takes a list of pattern, a list of corresponding desired outputs, min mean squared error,
        # max number of iterations, eta and momentum

        # set iterations to 0 and mean squared error to 100 to start while loop
        iterations = 0
        mse = 100.0

        while (mse > mse_min) and (iterations < max_iterations):
            # while loop loops until mse is below threshold or max iterations is reached

            mse = 0.0
            iterations += 1

            for i, pattern in enumerate(pattern_list):
                # go through all patterns in the pattern list, impose on input layer, and forward propagate

                self.init_input(pattern)

                self.forward_prop()

                for k in reversed(range(len(self.layer_list))):
                    # go backward through layers and update deltas

                    if k == len(self.layer_list) - 1:
                        # if output layer, calculate delta for nodes based on desired output

                        for l, node in enumerate(self.layer_list[k]):

                            node.delta = node.act * (1 - node.act) * (desired_list[i][l] - node.act)

                            # calculate sum of differences between desired and observed output
                            mse += pow((desired_list[i][l] - node.act), 2)

                    elif k > 0:

                        # if not output layer and not input layer (k == 0), calculate delta based on layer above
                        for l, node in enumerate(self.layer_list[k]):

                            node.hidden_node_error()

                            node.delta = node.act * (1 - node.act) * node.error

                for weight in self.weight_list:
                    # update delta weights based on delta (eta term used to avoid overshooting local min)

                    weight.delta_weight += weight.output_node.delta * weight.input_node.act * eta

            for weight in self.weight_list:
                # update weights from delta_weights

                weight.weight += weight.delta_weight

                weight.delta_weight *= momentum  # partially reset delta_weights based on momentum term

            # get mean squared error by dividing sum of squared errors by number of output nodes
            mse = mse / len(self.layer_list[-1])

            # print iteration and mse
            print('Iteration =', iterations, '; MSE =', mse)

    def display_nets(self):

        # This function prints layer inputs and activations as well as weights

        for j, layer in enumerate(reversed(self.layer_list)):

            if layer[0].node_type == 'output':

                print('## LAYER', str(j + 1), '##\n')
                for i, node in enumerate(layer):
                    print('Node', str(i + 1) + ':', 'Act=', node.act, ',Inp=', node.input, '	',)
                print('')

                for i, node in enumerate(layer):
                    print('Node', str(i + 1), 'weights')

                    for weight in node.weights_below:
                        print(weight.weight, ',',)
                    print('\n')
            elif layer[0].node_type == 'hidden':

                print('## LAYER', str(j + 1), '##\n')
                for i, node in enumerate(layer):
                    print('Node', str(i + 1) + ':', 'Act=', node.act, ',Inp=', node.input, '	', )
                print('')

                for i, node in enumerate(layer):
                    print('Node', str(i + 1), 'weights')

                    for weight in node.weights_below:
                        print(weight.weight, ',', )
                    print('\n')
            else:
                print('## LAYER', str(j + 1), '##\n')
                for i, node in enumerate(layer):
                    print('Node', str(i + 1) + ':', 'Act=', node.act, ',Inp=', node.input, '	',)
                print('')

    def test_net(self, test_patterns):

        # Imposes test patterns, forward propagates and displays network
        for i, pattern in enumerate(test_patterns):
            for layer in self.layer_list:
                for node in layer:
                    node.act = 0.0
            print('Pattern:', str(i + 1))
            self.init_input(pattern)
            self.forward_prop()
            self.display_nets()


# 3 different networks, uncomment for relevant modelling

# Network 1
# Create network taking values: num_layers, nodes_hidden, nodes_inp, nodes_out
# net = Network(2, 5, 10, 4)
# net.create_layers()
# net.create_weights()
#
# # Network 1 input and output patterns
# training_pattern1 = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
#
# output_pattern1 = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
#                    [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
#
# # Run back_prop taking: pattern_list, desired_list, mse_min, max_iterations, eta, momentum
# net.back_prop(training_pattern1, output_pattern1, 0.01, 10000, 0.3, 0.7)
#
# print('\n')
# net.test_net(training_pattern1)


# Network 2
# net = Network(2, 5, 8, 7)
# net.create_layers()
# net.create_weights()
#
# training_pattern2 = [[1, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1],
#                      [0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1]]
#
# output_pattern2 = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]]
#
# # Run back_prop taking: pattern_list, desired_list, mse_min, max_iterations, eta, momentum
# net.back_prop(training_pattern2, output_pattern2, 0.01, 10000, 0.3, 0.7)
#
# net.init_input([0, 0, 0, 0, 0, 0, 1, 1])
# net.forward_prop()
#
# print('\n')
#
# # Test on
# test_pattern = [[0, 0, 0, 0, 0, 0, 1, 1]]
#
# net.test_net(test_pattern)

# Network 3 - keras

# # set numpy printing to 1 decimal
# np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
#
# # Shut down Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# # Set random seed
# np.random.seed(42)
#
# input_data3 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                         [0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                         [0, 1, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
#
# output_data3 = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
#                          [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
#
# # Model creation
# model = Sequential()
#
# model.add(Dense(units=5, activation='relu', input_dim=10))
# model.add(Dense(units=5, activation='relu'))
# model.add(Dense(units=4, activation='sigmoid'))
#
# adam_opt = Adam(lr=0.0001)
# model.compile(loss='mean_squared_error', optimizer=adam_opt, metrics=['accuracy'])
#
# model.summary()
#
# # Model training
# model.fit(input_data3, output_data3, epochs=10000, batch_size=8, verbose=2)
#
# # Model test
# test_loss, test_accuracy = model.evaluate(x=input_data3, y=output_data3)
#
# print('test_loss:', test_loss, 'test_accuracy:', test_accuracy)
