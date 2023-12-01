from mnist import MNIST
import numpy as np

mndata = MNIST('./../../python-mnist/data')
images, labels = mndata.load_training()

PIXELS = 784
ALPHA = 0.1
PER_DESCENT = 100
DESCENTS = 50

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
bias_1 = np.random.rand(16) - 0.5
weights_1_2 = np.random.rand(16, 16) - 0.5
bias_2 = np.random.rand(16) - 0.5
weights_2_output = np.random.rand(10, 16) - 0.5
bias_output = np.random.rand(10) - 0.5

# these are the nodes / neurons in the neural network
input_neurons = np.random.rand(784)
layer_1_neurons = np.random.rand(16)
layer_2_neurons = np.random.rand(16)
output_neurons = np.random.rand(10)

def random_weight_bias():
    weights_input_1 = np.random.rand(16, PIXELS) - 0.5
    bias_1 = np.random.rand(16) - 0.5
    weights_1_2 = np.random.rand(16, 16) - 0.5
    bias_2 = np.random.rand(16) - 0.5
    weights_2_output = np.random.rand(10, 16) - 0.5
    bias_output = np.random.rand(10) - 0.5

def sigmoid(z):
    # trying to fix overflow with rearranged sigmoid depending on sign
    if z < 0:
        return (np.exp(z) / 1 + np.exp(z)) # slightly unsure of the correct of this
    return (1 / (1 + np.exp(-z))) 

def sigmoid_prime(z):
    val = sigmoid(z) * (1 - sigmoid(z))
    return val

def forward_propagation(testcase):
    input_neurons = testcase

    for i in range(len(layer_1_neurons)):
        z = input_neurons.dot(weights_input_1[i]) + bias_1[i]
        layer_1_neurons[i] = sigmoid(z)

    for i in range(len(layer_2_neurons)):
        z = layer_1_neurons.dot(weights_1_2[i]) + bias_2[i]
        layer_2_neurons[i] = sigmoid(z)
         
    for i in range(len(output_neurons)):
        z = layer_2_neurons.dot(weights_2_output[i]) + bias_output[i]
        output_neurons[i] = sigmoid(z)

def get_output():
    return output_neurons.argmax()

#cost for 1 number
def cost(ind):
    c = 0
    for i in range(10):
        val = 0
        if i == ind:
            val = 1
        c += (val - output_neurons[i])**2
    return c

"""
def backward_propagation():
"""

def test_NN(iterations):
    total = 0
    correct = 0
    for _ in range(iterations):
        ind = np.random.randint(len(test_data))
        forward_propagation(test_data[ind])
        if get_output() == test_labels[ind]:
            correct += 1
        total += 1
    print("Accuracy:", total/correct)

def gradient_descent():
    changes_weights_input_1 = np.zeros((16, PIXELS))
    changes_bias_1 = np.zeros(16)
    changes_weights_1_2 = np.zeros((16, 16))
    changes_bias_2 = np.zeros(16)
    changes_weights_2_output = np.zeros((10, 16))
    changes_bias_output = np.zeros(10)
     
    for _ in range(PER_DESCENT):
        ind = np.random.randint(len(train_data)) # the index of value you want to give the NN to help itself with
        forward_propagation(train_data[ind])
        optimal = np.zeros(len(output_neurons))
        Z = np.zeros(len(output_neurons))

        for i in range(len(optimal)):
            if i == ind:
                optimal[i] = 1
            else:
                optimal[i] = 0

        cost_vector = np.zeros(len(output_neurons))
        for i in range(len(output_neurons)):
            cost_vector[i] = output_neurons[i] - optimal[i];

        for i in range(len(output_neurons)):
            Z[i] = layer_2_neurons.dot(weights_2_output[i]) + bias_output[i]

        # TODO: remember to average out the additions on the 6 vectors above
    
        # adding cost-weight derivatives for last layer
        for i in range(len(output_neurons)):
            for j in range(len(changes_weights_2_output)):
                val = layer_2_neurons[j]
                val *= sigmoid_prime(Z[i])
                val *= 2 * cost_vector[i]
                changes_weights_2_output[i][j] += val

        # adding cost-bias derivatives for last layer
        for i in range(len(output_neurons)):
            val = 1
            val *= sigmoid_prime(Z[i])
            val *= 2 * cost_vector[i]
            changes_bias_output[j] += val

        optimal = np.zeros(len(layer_2_neurons))
        for i in range(len(layer_2_neurons)):
            sum = 0
            for j in range(len(output_neurons)):
                val = weights_2_output[j][i] # TODO: check this
                val *= sigmoid_prime(Z[j])
                val *= 2 * cost_vector[j]
                sum += val
            optimal[i] = sum

        ################################################################

        cost_vector = np.zeros(len(layer_2_neurons))
        for i in range(len(layer_2_neurons)):
            cost_vector[i] = layer_2_neurons[i] - optimal[i];

        Z = np.zeros(len(layer_2_neurons))
        for i in range(len(layer_2_neurons)):
            Z[i] = layer_1_neurons.dot(weights_1_2[i]) + bias_2[i]

        # adding cost-weight derivatives for 2nd hidden layer
        for i in range(len(layer_2_neurons)):
            for j in range(len(changes_weights_1_2)):
                val = layer_1_neurons[j]
                val *= sigmoid_prime(Z[i])
                val *= 2 * cost_vector[i]
                changes_weights_1_2[i][j] += val

        # adding cost-bias derivatives for 2nd hidden layer
        for i in range(len(layer_2_neurons)):
            val = 1
            val *= sigmoid_prime(Z[i])
            val *= 2 * cost_vector[i]
            changes_bias_2[j] += val

        optimal = np.zeros(len(layer_1_neurons))
        for i in range(len(layer_1_neurons)):
            sum = 0
            for j in range(len(layer_2_neurons)): 
                val = weights_1_2[j][i] # TODO: check this j i nonsense
                val *= sigmoid_prime(Z[j])
                val *= 2 * cost_vector[j]
                sum += val
            optimal[i] = sum
        
        ################################################################

        cost_vector = np.zeros(len(layer_1_neurons))
        for i in range(len(layer_1_neurons)):
            cost_vector[i] = layer_1_neurons[i] - optimal[i];

        Z = np.zeros(len(layer_1_neurons))
        for i in range(len(layer_1_neurons)):
            Z[i] = input_neurons.dot(weights_input_1[i]) + bias_1[i]

        # adding cost-weight derivatives for 1st hidden layer
        for i in range(len(layer_1_neurons)):
            for j in range(len(changes_weights_input_1)):
                val = input_neurons[j]
                val *= sigmoid_prime(Z[i])
                val *= 2 * cost_vector[i]
                changes_weights_input_1[i][j] += val

        # adding cost-bias derivatives for 1st hidden layer
        for i in range(len(layer_1_neurons)):
            val = 1
            val *= sigmoid_prime(Z[i])
            val *= 2 * cost_vector[i]
            changes_bias_1[j] += val

    changes_weights_input_1 = np.divide(changes_weights_input_1, PER_DESCENT)
    changes_bias_1 = np.divide(changes_bias_1, PER_DESCENT)
    changes_weights_1_2 = np.divide(changes_weights_1_2, PER_DESCENT)
    changes_bias_2 = np.divide(changes_bias_2, PER_DESCENT)
    changes_weights_2_output = np.divide(changes_weights_2_output, PER_DESCENT)
    changes_bias_output = np.divide(changes_bias_output, PER_DESCENT)
    
    # apply gradients to actual weights and biases
    for i in range(16):
        for j in range(PIXELS):
            weights_input_1[i][j] += changes_weights_input_1[i][j] * ALPHA
    for i in range(16):
        bias_1[i] += changes_bias_1[i] * ALPHA

    for i in range(16):
        for j in range(16):
            weights_1_2[i][j] += changes_weights_1_2[i][j] * ALPHA
    for i in range(16):
        bias_2[i] += changes_bias_2[i] * ALPHA

    for i in range(16):
        for j in range(10):
            weights_input_1[i][j] += changes_weights_input_1[i][j] * ALPHA
    for i in range(10):
        bias_output[i] += changes_bias_output[i] * ALPHA

def brrr():
    for _ in range(DESCENTS):
        gradient_descent()
        test_NN(PER_DESCENT) # <- no specific reason for setting this equal to the PER_DESCENT
brrr()

# print(weights_1_2)
# print(bias_2)
# print(weights_2_output)
# print(bias_output)
