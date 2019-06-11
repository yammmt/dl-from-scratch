# pp. 110-111

import numpy as np
import os
import sys
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # Gaussian distribution

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print('net.W:')
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print('p: {}'.format(p))
print('np.argmax(p): {}'.format(np.argmax(p)))

t = np.array([0, 0, 1]) # answer label
print('net.loss(x, t): {}'.format(net.loss(x, t)))

f = lambda _w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print('dW: {}'.format(dW))
