import numpy as np
import math as math
import random as random
import os as os
from PIL import Image


class BPNeuralNetwork(object):
    def __init__(self, nodes_list, learning_rate, activation_function, activation_function_derivative,
                 output_activation_function, output_activation_function_derivative):
        # set nodes in layers
        self.nodes_list = nodes_list

        # initialize weight matrix and bias matrix
        self.weight_matrix_list = []
        self.bias_vector_list = []
        for index in range(1, len(nodes_list)):
            weight_matrix = np.random.normal(0.0, 0.0004,
                                             (self.nodes_list[index], self.nodes_list[index - 1]))
            self.weight_matrix_list.append(weight_matrix)
            bias_vector = np.random.normal(0.0, 0.0004, (self.nodes_list[index], 1))
            self.bias_vector_list.append(bias_vector)

        # initialize learning rate
        self.learning_rate = learning_rate

        # initialize activation_function and derivative
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.output_activation_function = output_activation_function
        self.output_activation_function_derivative = output_activation_function_derivative

    def forward(self, input_vector):
        # make sure the length of inputs valid
        assert len(input_vector) == self.nodes_list[0]

        # make the output tuple list
        output_tuple_list = []

        # forward inputs to get actual outputs
        current_output_vector = input_vector
        for i in range(0, len(self.weight_matrix_list)):
            weight_matrix = self.weight_matrix_list[i]
            bias_vector = self.bias_vector_list[i]
            current_input_vector = np.dot(weight_matrix, current_output_vector) + bias_vector
            if i == len(self.weight_matrix_list) - 1:
                current_output_vector = self.output_activation_function(current_input_vector)
            else:
                current_output_vector = self.activation_function(current_input_vector)
            output_tuple_list.append((current_input_vector, current_output_vector))

        # return the forward output_tuple_list
        return output_tuple_list

    def cal_gradient_for_regression(self, inputs, desired_outputs):
        # cost function |desired_output - actual_output|

        # convert to vector
        input_vector = np.array(inputs, ndmin=2).T
        desired_output_vector = np.array(desired_outputs, ndmin=2).T

        # make a gradient list
        gradient_list = []
        weight_gradient_list = []
        bias_gradient_list = []

        # forward inputs to get actual outputs
        actual_output_tuple_list = self.forward(input_vector)

        # calculate output_layer gradient
        actual_output_vector_z = actual_output_tuple_list[-1][0]
        actual_output_vector_a = actual_output_tuple_list[-1][1]
        output_gradient = (actual_output_vector_a - desired_output_vector) * self.output_activation_function_derivative(
            actual_output_vector_z)
        gradient_list.insert(0, output_gradient)

        # calculate loss
        loss = np.sum(np.square(desired_output_vector - actual_output_vector_a)) / 2

        # calculate output_layer bias and weight gradient
        last_layer_output_vector_a = actual_output_tuple_list[-2][1]
        weight_gradient_list.insert(0, np.dot(output_gradient, last_layer_output_vector_a.T))
        bias_gradient_list.insert(0, output_gradient)

        # calculate hidden_layer gradient
        for i in range(len(self.weight_matrix_list) - 1, 0, -1):
            actual_output_vector_z = actual_output_tuple_list[i - 1][0]
            hidden_gradient = np.dot(self.weight_matrix_list[i].T,
                                     gradient_list[0]) * self.activation_function_derivative(actual_output_vector_z)
            gradient_list.insert(0, hidden_gradient)

            # calculate hidden_layer bias and weight gradient
            if i >= 2:
                last_layer_output_vector_a = actual_output_tuple_list[i - 2][1]
            else:
                last_layer_output_vector_a = input_vector
            weight_gradient_list.insert(0, np.dot(hidden_gradient, last_layer_output_vector_a.T))
            bias_gradient_list.insert(0, hidden_gradient)

        return weight_gradient_list, bias_gradient_list, loss

    def update(self, weight_gradient_list, bias_gradient_list):
        for i in range(0, len(self.weight_matrix_list)):
            self.weight_matrix_list[i] -= weight_gradient_list[i] * self.learning_rate
            self.bias_vector_list[i] -= bias_gradient_list[i] * self.learning_rate


def sigmoid(vec):
    return 1 / (1 + np.exp(-vec))


def sigmoid_derivative(vec):
    m = sigmoid(vec)
    return np.dot(m, (1 - m).T) * np.eye(m.shape[0])

def softmax(vec):
    return np.exp(vec) / np.sum(np.exp(vec))

def softmax_derivative(vec):
    m = softmax(vec)
    return np.dot(m, m.T) + m * m


def identity(mat):
    return mat

def identity_derivative(mat):
    return np.ones(mat.shape)

def square_error_loss(d, o):
    return np.sum(np.square(d - o)) / 2

def square_error_loss_derivative(d, o):
    return o - d

def cross_entropy_loss(d, o):
    return -np.sum(d * np.log(o))

def cross_entropy_loss_derivative(d, o):


def task1():
    node_list = [1, 50, 50, 1]
    lr = 0.075
    bpnn = BPNeuralNetwork(node_list, lr, sigmoid, sigmoid_derivative, identity, identity_derivative)

    print("-----before training-----")
    print("weight_matrix_list ")
    print(bpnn.weight_matrix_list)
    print("bias_vector_list ")
    print(bpnn.bias_vector_list)
    print("-------in training-------")

    total_error = 0
    for j in range(0, 1000000):
        x = random.random() * 2 * math.pi - math.pi
        y = math.sin(x)
        weight_gradients, bias_gradients, error = bpnn.cal_gradient_for_regression([x], [y])
        total_error += error
        if j % 10000 == 0:
            print("loss:%.6f" % (total_error / 10000))
            total_error = 0
        bpnn.update(weight_gradients, bias_gradients)


def task2():
    # load bmp file to array list
    test_list = []
    for i in range(1, 15):
        for j in range(0, 256):
            im = Image.open('TRAIN/%d/%d.bmp' % (i, j), mode='r')
            rgb_im = im.convert('1')
            inputs = []
            for k in range(0, rgb_im.size[0]):
                row = [rgb_im.getpixel((k, l)) / 255 for l in range(0, rgb_im.size[1])]
                inputs.extend(row)
            outputs = [0.0 for r in range(1, 15)]
            outputs[i - 1] = 1
            test_list.append((inputs, outputs))

    # start training
    lr = 0.075

    for i in range(0, 10000):
        # shuffle the list
        random.shuffle(test_list)



    # im = Image.open('TRAIN/1/0.bmp', mode='r')
    # rgb_im = im.convert('1')
    # print(rgb_im)
    #
    # print([[rgb_im.getpixel((i, j)) / 255 for j in range(0, rgb_im.size[1])] for i in range(0, rgb_im.size[1])])


# # set nodes in layers
# self.input_nodes = input_nodes
# self.hidden_nodes_list = hidden_nodes_list
# self.output_nodes = output_nodes
#
#
# self.weight_matrix_hidden_list = []
# self.weight_matrix_output = np.random.normal(0.0, self.output_nodes ** -0.5, ())
# # initialize weight matrix
# self.weight_matrix_input2hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
#                                                    (self.hidden_nodes, self.input_nodes))
# self.weight_matrix_hidden2output = np.random.normal(0.0, self.output_nodes ** -0.5,
#                                                     (self.output_nodes, self.hidden_nodes))
# # initialize learning rate
# self.learning_rate = learning_rate
#
# # initialize activation function with sigmoid
# self.activation_function = (lambda x: 1 / (1 + np.exp(-x)))

# def train(self, actual_input_list, actual_output_list):
#     input_vector = np.array(actual_input_list, ndmin=2).T
#     actual_output_vector = np.array(actual_output_list, ndmin=2).T
#
#     hidden_in_vector = np.dot(self.weight_matrix_input2hidden, input_vector)
#     hidden_out_vector = self.activation_function(hidden_in_vector)
#
#     output_in_vector = np.dot(self.weight_matrix_hidden2output, hidden_out_vector)
#     output_out_vector = output_in_vector
#
#     # get the error
#     output_error = np.subtract(output_out_vector, actual_output_vector)
