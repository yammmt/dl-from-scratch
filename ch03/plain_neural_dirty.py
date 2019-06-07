# 3.4.2 pp. 60-64

import numpy as np

def identity_function(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([1.0, 0.5])

print('--- first layer ---')
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print('W1.shape: {}'.format(W1.shape))
print('X.shape: {}'.format(X.shape))
print('B1.shape: {}'.format(B1.shape))

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print('A1: {}'.format(A1))
print('Z1: {}'.format(Z1))


print('\n--- second layer ---')
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print('Z1.shape: {}'.format(Z1.shape))
print('W2.shape: {}'.format(W2.shape))
print('B2.shape: {}'.format(B2.shape))

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print('A2: {}'.format(A2))
print('Z2: {}'.format(Z2))


print('\n--- third layer ---')
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
print('Z2.shape: {}'.format(Z2.shape))
print('W3.shape: {}'.format(W3.shape))
print('B3.shape: {}'.format(B3.shape))

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print('A3: {}'.format(A3))
print('Y: {}'.format(Y))
