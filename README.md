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




