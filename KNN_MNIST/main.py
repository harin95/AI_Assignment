import sys, os
sys.path.append(os.pardir)
import KNNclass as knn
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n_class = len(label_name)

# load data
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

k = 10

test = knn.KNN(k, x_train, y_train, x_test, y_test)

#hand crafted features & the number of data
x2_train, n_train = test.hand_crafted(x_train)
x2_test, n_test = test.hand_crafted(x_test)
s = x2_train[0].size # number of input features

#random sample data
size = 1000
sample = np.random.randint(0, x2_test.shape[0], size)
n_sample = sample.size


print("----------hand crafted feature----------")
print('k = ', k)
acc2 = np.zeros(n_sample, bool)
for j in sample:
    distance2 = test.cal_d(x2_test[j], x2_train, n_train) #calculate distance
    s_idx2, s_distance2 = test.k_nearest(k, distance2) #sorting and slicing
    test.weighted_majority_vote(s_idx2, s_distance2, y_train, k, n_class, j, y_test, sample, acc2)

accuracy = test.cal_accuracy(acc2)
print("accuracy: ", accuracy)


