#!/usr/bin/env python
# coding: utf-8

# In[114]:


# Import the NumPy library
import numpy as np


# In[115]:


#  Function to normalize data in the range [0,1] using min-max feature scaling
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Input data representing height and weight of patients
X = np.array([[161,  55],
              [183,  95],
              [171,  60],
              [194, 102],
              [162,  58],
              [185,  90],
             ])

# Normalize the input data
scaled_x = NormalizeData(X)

# Print the scaled input data
print(scaled_x)


# In[116]:


# Sigmoid activation function : f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[117]:


# Derivative of the sigmoid function : f'(x) = f(x) * (1 - f(x))
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# In[118]:


# Mean squared error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


# In[119]:


# Neural network class
class OurNeuralNetwork:
    def __init__(self):
        # Initialize random weights and biases
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = 1
        self.b2 = 1
        self.b3 = 1
        
    def feedforward(self, x):
        # Forward pass through the network
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        
        for epoch in range(epochs):
            for x,y_true in zip(data, all_y_trues):
                # Forward pass
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Backpropagation
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)   
                
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # Update the weights and biases
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                
                # Compute and print the loss every 10 epochs
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))


# In[120]:


# Define the input data and corresponding target values
data = np.array([[0.76258993, 0.00000000],
                 [0.92086331, 0.28776978],
                 [0.83453237, 0.03597122],
                 [1.00000000, 0.3381295 ],
                 [0.93525180, 0.25179856],
                ])

all_y_trues = np.array([0,
                        1,
                        0,
                        1,
                        1,
                       ])


# In[121]:


# Create an instance of the OurNeuralNetwork class
network = OurNeuralNetwork()

# Train the neural network using the given input data and target values
network.train(data, all_y_trues)


# In[122]:


# Create an array representing the features (height and weight) of the fifth user
name5 = np.array([0.76978417, 0.02158273])

# Feed the features of the fifth user to the trained network and print the output prediction
print("Name 5: %.3f" % network.feedforward(name5))

