# pp. 74-75

import os
import sys
import numpy as np
from PIL import Image
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print('label: {}'.format(label))

print('img.shape: {}'.format(img.shape))
img= img.reshape(28, 28)
print('img.shape: {}'.format(img.shape))

img_show(img)
