from mnist import MNIST
import numpy as np

mndata = MNIST('./../../python-mnist/data')
images, labels = mndata.load_training()

PIXELS = 784

# print(images[0])
# print(type(images))

images = np.array(images)
labels = np.array(labels)

n , _ = images.shape

print("labels:", len(labels))
train_data = images[1:(n-9999)]
test_data = images[(n-10000):n]
train_labels = labels[1:(n-9999)]
test_labels = labels[(n-10000):n]
# print("train_labels:", len(train_labels))
# print("test_labels:", len(test_labels))

# will have 2 hidden layers because 3b1b does :P

# will use sigmoid, so weights and biases should be in ~[-0.5, 0.5]
weights_input_1 = np.random.rand(16, PIXELS) - 0.5
bias_input_1 = np.random.rand(16, 1) - 0.5
weights_1_2 = np.random.rand(16, 16) - 0.5
bias_1_2 = np.random.rand(16, 1) - 0.5
weights_2_output = np.random.rand(10, 16) - 0.5
bias_2_output = np.random.rand(10, 1) - 0.5

# these are the nodes / neurons in the neural network
input_neurons = np.random.rand(784)
layer_1_neurons = np.random.rand(16)
layer_2_neurons = np.random.rand(16)
output_neurons = np.random.rand(10)
print("layer_1_neurons:",layer_1_neurons)

def random_weight_bias():
    weights_input_1 = np.random.rand(16, PIXELS) - 0.5
    bias_input_1 = np.random.rand(16, 1) - 0.5
    weights_1_2 = np.random.rand(16, 16) - 0.5
    bias_1_2 = np.random.rand(16, 1) - 0.5
    weights_2_output = np.random.rand(10, 16) - 0.5
    bias_2_output = np.random.rand(10, 1) - 0.5

def sigmoid(z):
    return (1 / (1 + np.exp(-z))) 

def forward_propagation(testcase):
    input_neurons = testcase

    for i in range(len(layer_1_neurons)):
        z = input_neurons.dot(weights_input_1[i]) + bias_input_1[i]
        layer_1_neurons[i] = sigmoid(z)

    for i in range(len(layer_2_neurons)):
        z = layer_1_neurons.dot(weights_1_2[i]) + bias_1_2[i]
        layer_2_neurons[i] = sigmoid(z)
         
    for i in range(len(output_neurons)):
        z = layer_2_neurons.dot(weights_2_output[i]) + bias_2_output[i]
        output_neurons[i] = sigmoid(z)

def get_output():
    return output_neurons.argmax() + 1

"""
def test_NN():
    total = 0
    correct = 0
"""
