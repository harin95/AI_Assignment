import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import KNNclass as KNN

iris = load_iris()

X = iris.data
y = iris.target
y_name = iris.target_names

#data partition for training and testing
list_a = list(range(150))
list_b = list(range(14, 150, 15))

X_training = np.empty((0,4), float) #initialize
X_testing = np.empty((0,4), float)

for i in list_a:
    if i not in list_b:
        X_training = np.append(X_training, [X[i, :]], axis=0)
    if i in list_b:
        X_testing = np.append(X_testing, [X[i, :]], axis=0)

#target partition
y_training = np.empty(0, int)
y_testing = np.empty(0, int)

for i in list_a:
    if i not in list_b:
        y_training = np.append(y_training, y[i])
    if i in list_b:
        y_testing = np.append(y_testing, y[i])


'''--------------------------------'''

k_list = [3, 5, 10]

for k in k_list:
    print("k value = ", k)

    test_k = KNN.KNN(k, X_training, X_testing, y_training, y_testing)
    distance = test_k.cal_d(X_training, X_testing)
    kn, distance = test_k.k_nearest(k, distance)
    majority_vote = test_k.majority_vote(k, kn, y_training)
    test_k.print_result(majority_vote, y_testing, y_name, 0)
    print("\n")

    weighted_vote = test_k.weighted_majority_vote(k, kn, distance, y_training)
    test_k.print_result(weighted_vote, y_testing, y_name, 1)
    print("\n\n")

'''---------------------------------'''



