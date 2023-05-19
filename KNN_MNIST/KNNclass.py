import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

class KNN:
    def __init__(self, k, x1, y1, x2, y2):
        self.k = k
        #training
        self.x1 = x1
        self.y1 = y1
        #testing
        self.x2 = x2
        self.y2 = y2


    def hand_crafted(self, x1):
        s = int(x1.size/x1[0].size) # 데이터 갯수
        xx = x1.reshape(s, 28, 28)
        y = xx > 0  # feature값 0이 아닌것을 골라내서 갯수 셈
        y2 = np.sum(y, axis = 2)    #행
        y3 = np.sum(y, axis = 1)    #열
        y4 = np.append(y2, y3, axis = 1)  #1차원 배열로 만듦
        return y4, s


    def cal_d(self, x, x2, n):  # x: test input,    x2: training data
        distance = np.zeros(n)
        for i in range(n):
            d = np.sqrt(np.sum((x2[i]-x)**2))
            distance[i] = d
        return distance


    def k_nearest(self, k, distance):
        sorted_idx = distance.argsort()
        s_idx = sorted_idx[:k]  #거리 오름차순으로 인덱스 정렬
        distance.sort() #majority vote위해서 거리 오름차순으로 정렬
        s_distance = distance[:k]
        return s_idx, s_distance


    def weighted_majority_vote(self, s_idx, s_distance, y_train, k, n_class, j, y_test, sample, acc):
        l = []  #l : 뽑은 k개의 training data class
        vote = np.zeros(n_class, float)
        w = 1/s_distance

        for i in range(k):
            l.append(y_train[s_idx[i]]) #training data class 구하기

        for i in range(k):    #class마다 weight계산
            vote[l[i]] += w[i]

        result = np.argmax(vote) #max값 구해서 실제값과 비교
        label = y_test[j]

        Idx = np.where(sample == j) #출력인덱스
        # 결과 출력
        print(j, "th data\tresult: ", result, "label: ", label, '\t\t[', Idx[0][0], ']')

        if(result == label): #결과 일치할때마다 true로 변경
            acc[np.where(sample == j)] = True


    def cal_accuracy(self, x):
        return np.sum(x)/x.size

