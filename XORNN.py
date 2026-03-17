import numpy as np 

# Data
x = np.array([[0,0],[1,1],[0,1],[1,0]]) #(4,2)
y = np.array([[0],[0],[1],[1]])         #(4,1)

#Activation Function (map everything between 0 and 1 center around 0.5 (probability))
#Whereas is centerend around 0 (between -1 and 1) makes learning for the next layer a little bit easier

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s*(1 - s)

# Weights and bias initialization
np.random.seed(42) #generates a new set or repeats pseudo-random numbers
W1 = np.random.randn(2,4)* 0.1 # weights of first layer : (2 input -> 4 neurons =  4 outputs) an array of of 2 element inside of which there is 4 element
b1 = np.zeros((1,4))           # bias of first layer 
W2 = np.random.randn(4,1)* 0.1 # weights of second layer : (4 inmputs -> 1 output)
b2 = np.zeros((1,1))

# training loop
lr = 0.5 # learning rate
epochs = 5000

for epoch in range(epochs):
    # Forward pass 
    Z1 = x @ W1 + b1          # (4, 4)  : layer 1 linear combination (matricial product)
    A1 = sigmoid(Z1)           # (4, 4)  : activate layer 1
    Z2 = A1 @ W2 + b2          # (4, 1)  : layer 2 linear combination
    A2 = sigmoid(Z2)           # (4, 1)  : final prediction

    # Compute loss (mse)
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))

    # Backward (backpropagation)
    dA2 = 2 * (A2 - y) / y.shape[0] # gradient of loss
    dZ2 = A2 -y   # gradient before W2 

    dW2 = A1.T @ dZ2                # .T est la transposé de la matrice            
    db2 = np.sum(dZ2, axis=0, keepdims=True) 

    dA1 = dZ2 @ W2.T         # go up the first layer
    dZ1 = dA1 * sigmoid_deriv(Z1)

    dW1 = x.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True) # axis = 0 sum on lines (vertically)

    #Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")



print("\nPrédictions finales :")
print(np.round(A2, 3))
print("Attendu :", y.T)