import numpy as np

# Input dataset (XOR gate)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Target output
y = np.array([[0],[1],[1],[0]])

# Sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x*(1-x)

# Initialize weights
np.random.seed(1)

input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

learning_rate = 0.1
epochs = 10000

for i in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    predicted_output = sigmoid(final_input)

    # Error
    error = y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Updating weights
    W2 += hidden_output.T.dot(d_predicted_output) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate

print("Predicted Output:")
print(predicted_output)