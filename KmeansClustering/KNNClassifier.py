import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class KNN:
    dist_dict = defaultdict(list)

    # Converting imgae data into CSV
    def convert(self, imgf, labelf, outf, n):
        f = open(imgf, "rb")
        o = open(outf, "w")
        l = open(labelf, "rb")

        f.read(16)
        l.read(8)
        images = []

        for i in range(n):
            image = [ord(l.read(1))]
            for j in range(28 * 28):
                image.append(ord(f.read(1)))
            images.append(image)

        for image in images:
            o.write(",".join(str(pix) for pix in image) + "\n")
        f.close()
        o.close()
        l.close()

    # convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "mnist_train.csv", 6000)
    # convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "mnist_test.csv", 1000)

    def loadData(self):
        trainingData = np.genfromtxt('mnist_train.csv', delimiter=',', dtype='int')
        test_data = np.genfromtxt('mnist_test.csv', delimiter=',', dtype='int')
        return trainingData, test_data

    def count_max_labels_of_cluster(self, nearest_cluster):
        labels = [x[0] for x in nearest_cluster]
        digit = np.bincount(np.array(labels)).argmax()
        return digit

    def runKNN(self, train_data, test_data, K):
        incorrect = [0] * len(K)

        for data in test_data:
            dist = self.getDistance(data, train_data)
            for i in range(len(K)):
                if (data[0] != self.count_max_labels_of_cluster(dist[:K[i]])):
                    incorrect[i] += 1
        length = len(test_data)
        error_rate = [x / length for x in incorrect]
        # self.plotGraph(K, error_rate)

        print("incorrect metric is", error_rate)

    def getDistance(self, test_instance, train_data):

        test_arr = np.array(test_instance)
        train_arr = np.array(train_data)
        # To substract it from all the target points in constant time
        sub = test_arr[1:] - train_arr[:, 1:]
        label = train_arr[:, 0]
        # To make it euclidean distance
        sqr = np.square(sub)
        dist = np.sum(sqr, axis=1)
        # To append the label back to the array
        dist = np.column_stack((label, dist))
        dist = dist[np.argsort(dist[:, 1])]
        return dist

    def plotGraph(self, x_values, y_values):
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        fig.suptitle(' KNN classifier ')
        ax.set_xlabel(' K values ')
        ax.set_ylabel(' Error rates ')
        plt.draw()
        plt.show()

start = time.time()
n = KNN()
k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
train_data, test_data = n.loadData()
load_data = n.runKNN(train_data, test_data, k)
end = time.time()
print("Total time for KNN classifier ", end - start)
