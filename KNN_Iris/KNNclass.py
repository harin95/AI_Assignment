import numpy as np
np.set_printoptions(threshold=np.inf)

class KNN:
    def __init__(self, k, X1, X2, y1, y2):
        self.k = k
        #training
        self.X1 = X1
        self.y1 = y1
        #testing
        self.X2 = X2
        self.y2 = y2



    def cal_d(self, X, X2):
        distance = np.empty((0, 140), float)
        for i in range(10):
            arr = X-X2[i, :]
            arr = arr**2
            arr = np.sum(arr, axis=1)
            arr = arr**0.5
            distance = np.append(distance, [arr], axis=0)

        return distance


    def k_nearest(self, k, distance):
        sorted_d = distance.argsort()
        kn = sorted_d[:, :k]  #index list from nearest to farthest
        distance = np.sort(distance) #distance sorted for weighted majority vote
        distance = distance[:, :k]
        return kn, distance

    def majority_vote(self, k, kn, y_training):
        vote = np.empty(0, int)
        for i in range(10):
            val_list = []
            for j in range(k):
                val1 = y_training[kn[i,j]]  #get the class data of training input
                val_list.append(val1)
            val_list = np.array(val_list)
            a1 = np.count_nonzero(val_list == 0)    #count class numbers
            a2 = np.count_nonzero(val_list == 1)
            a3 = np.count_nonzero(val_list == 2)
            arr = np.array([a1, a2, a3])
            result = arr.argmax()   #choose maximum
            vote = np.append(vote, result)
        return vote


    def weighted_majority_vote(self, k, kn, distance, y_training):
        weight = np.full((k), 10)
        vote = np.empty(0, int)
        for i in range(10):
            val_list = []
            weight = weight - distance[i, :]    #the nearer it is, the more weighted it has
            cls_list = np.zeros(3, int)

            for j in range(k):
                val1 = y_training[kn[i,j]]  #get the class data of training input
                val_list.append(val1)
            val_list = np.array(val_list)
            for j in range(k):
                if(val_list[j] == 0):
                    cls_list[0] += weight[j]    #calculate the weight for each class
                elif(val_list[j] == 1):
                    cls_list[1] += weight[j]
                elif(val_list[j] == 2):
                    cls_list[2] += weight[j]

            result = cls_list.argmax() #choose the maximum
            vote = np.append(vote, result)
        return vote



    def print_result(self, vote, y_testing, y_name, method):

        if(method == 0):
            print("<<<Majority vote>>>")
        elif(method == 1):
            print("<<<Weighted majority vote>>>")
        for i in range(10):
            if (y_name[vote[i]] == y_name[y_testing[i]]):
                print("test input index: ", i, "\tcomputed class", y_name[vote[i]], "\ttrue class: ", y_name[y_testing[i]])
            if(y_name[vote[i]] != y_name[y_testing[i]]):
                print("test input index: ", i, "\tcomputed class", y_name[vote[i]], "\ttrue class: ", y_name[y_testing[i]], "\t<<<<< not matching")

        print("Precision: ", np.mean(np.equal(y_testing, vote)))