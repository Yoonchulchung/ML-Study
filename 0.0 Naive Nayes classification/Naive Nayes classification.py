# Naive Nayes Classification 나이브 베이즈 분류: 가설이 실패여서 사용 x
# Optical character recognition

from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import mxnet as mx
from mxnet import nd
import numpy as np

#We go over one observation at a time (speed doesn't matter here)
def transform(data, label):
     return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)

mnist_train = mx.gluon.data.vision.MNIST(train = True, transform = transform)
mnist_test  = mx.gluon.data.vision.MNIST(train = False, transform = transform)

# Initialize the counters
xcount = nd.ones((784, 10))
ycount = nd.ones((10))

print("xcount: ", xcount)
print("ycount: ", ycount)

for data, label in mnist_train:
     y = int(label)
     ycount[y] += 1
     xcount[:, y] += data.reshape((784)) # data 값을 28 x 28 행렬로 놓으면 
                                         # 숫자고 보임.

# using broadcast again for division
py = ycount / ycount.sum()
px = ( xcount / ycount.reshape(1,10) )  # 특정 숫자 그림의 평균값.

fig, figarr = plt.subplots (1, 10, figsize = (10, 10))
for i in range(10):
     figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap = 'hot')
     figarr[i].axes.get_xaxis().set_visible(False)
     figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print('Class probabilities', py)

logpx = nd.log(px)
logpxneg = nd.log(1 - px)
logpy = nd.log(py)

def bayespost(data):
     # We need to incorporate the prior probability p(y) since p(y|x) is
     # proportional to p(x|y) p(y)
     logpost = logpy.copy()
     logpost += (logpx * data + logxneg * (1 - data)).sum(0)
     # Normalize to prevent overflow or underflow by subtracting the largest
     # value
     logpost -= nd.max(logpost)
     # Compute the softmax using logpx
     post = nd.exp(logpost).asnumpy()
     post /= np.sum(post)
     return post

fig, figarr = plt.subplots(2, 10, figsize=(10, 3))

# Show 10 images
ctr = 0
for data, label in mnist_test:
     x = data.reshape((784,1))
     y = int(label)

     post = bayespost(x)

     # Bar chart and image of digit
     figarr[1, ctr].bar(range(10), post)
     figarr[1, ctr].axes.get_yaxis().set_visible(False)
     figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
     figarr[0, ctr].axes.get_xaxis().set_visible(False)
     figarr[0, ctr].axes.get_yaxis().set_visible(False)
     ctr += 1

     if ctr == 10:
          break

plt.show()

# Initialize counter
ctr = 0
err = 0

for data, label in mnist_test:
    ctr += 1
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)
    if (post[y] < post.max()):
        err += 1

print('Naive Bayes has an error rate of', err/ctr)