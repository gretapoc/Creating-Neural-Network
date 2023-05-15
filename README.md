# Creating Neural Network


## Objective

Let's create a neural network that would predict the gender of the fifth user (0 - female, 1 - male) based on height and weight features. The height and weight attributes are the input data for the neural network, and the gender prediction is the output. 

![image]()

The data is given:

| Patient | Gender | Height | Weight |
| ------- | ------ | ------ | ------ |
| Name 1  | 0      | 161    | 55     |
| Name 2  | 1      | 183    | 95     |
| Name 3  | 0      | 171    | 60     |
| Name 4  | 1      | 194    | 102    |
| Name 5  | ?      | 162    | 58     |
| Name 6  | 1      | 185    | 90     |


In the network, we will:

- Utilize the sigmoid activation function and the backpropagation algorithm. 
- Perform training with different learning rates and numbers of cycles. The batch size is set to 1. 
- Generate initial weights randomly within the interval [-1,1]. 
- Normalize the data to the range [0,1] using min-max feature scaling.

## Solution

Importing the NumPy library.
```
# Import the NumPy library
import numpy as np
```

Normalize the data in the range [0,1] using min-max feature scaling.
```
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
```

Defining sigmoid activation function.
```
# Sigmoid activation function : f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

Defining derivative of the sigmoid function.
```
# Derivative of the sigmoid function : f'(x) = f(x) * (1 - f(x))
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)
```


Defining mean squared error loss function
```
# Mean squared error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
```


