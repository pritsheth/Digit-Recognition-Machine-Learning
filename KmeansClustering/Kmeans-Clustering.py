import random
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


class KNN:
    centroids = {}

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

        t_data = []
        for data in trainingData:
            t_data.append([data[0], data[1:]])
        # print(np.array(t_data,object))
        return t_data, test_data


    # Computing mean of the cluster to form the new mean
    def init_new_centroids(self, train_data, K):
        centroids = random.sample(train_data, K)
        return centroids

    def reform_cluster(self, centroids, train_data):
        length_of_centroids = len(centroids)

        cluster = [[] for i in range(length_of_centroids)]

        for point in train_data:
            min_dist = float("inf")
            centroid_index = 0
            for i in range(length_of_centroids):
                # norm_distance = np.linalg.norm(np.array(centroids[i][1:]) - np.array(point[1:]))
                norm_distance = distance.euclidean(centroids[i], point[1])

                if (norm_distance < min_dist):
                    min_dist = norm_distance
                    centroid_index = i

            # Appending to the closest centroid:
            cluster[centroid_index].append(point)

        return cluster

    def findMean(self, cluster):
        sliced_col = np.array(cluster, object)[:, 1]
        return np.mean(sliced_col, axis=0)

    def reform_unlabelled_centroids(self, clusters):
        unlabelled_centroids = []
        for c in clusters:
            unlabelled_centroids.append([self.findMean(c)])
        return unlabelled_centroids

    def repeat_until_convergence(self, centroids, clusters, train_data):

        old_centroid = centroids
        old_cluster = clusters

        while (True):

            new_centroid = self.reform_unlabelled_centroids(old_cluster)
            new_cluster = self.reform_cluster(new_centroid, train_data)

            diff = [distance.euclidean(o, n) for o, n in zip(old_centroid, new_centroid)]
            # print("difference between cetroid is ", diff)
            if diff.count(0.0) == len(clusters):
                break

            old_centroid = new_centroid
            old_cluster = new_cluster

        return new_centroid, new_cluster

    def count_max_labels_of_cluster(self, clusters):

        labels_centroid = []
        for cluster in clusters:
            labels = list(map(lambda x: x[0], cluster))
            digit = np.bincount(np.array(labels)).argmax()
            labels_centroid.append(digit)

        # print(" labelled cluster ", labels_centroid)
        return labels_centroid

    def predict_digit(self, test_data, centroids, labels):
        incorrect = 0
        for point in test_data:
            min_dist = float("INF")
            centroid_index = 0
            for i in range(len(centroids)):
                # norm_distance = np.linalg.norm(np.array(centroids[i]) - np.array(point[1:]))
                norm_distance = distance.euclidean(centroids[i], point[1:])

                if (norm_distance < min_dist):
                    min_dist = norm_distance
                    centroid_index = i

            if (point[0] != labels[centroid_index]):
                incorrect += 1

        print("Total incorrect predictions are ", incorrect)
        # print("Error rate is ", incorrect/len(test_data))
        pass

    def runKNN(self, train_data, test_data, K):
        print("Running KNN for K value: ", K)

        lab_centroids = self.init_new_centroids(train_data, K)
        centroids = [c[1] for c in lab_centroids]

        start1 = time.time()
        clusters = self.reform_cluster(centroids, train_data)
        end1 = time.time()
        print("time for reforming cluster ", end1 - start1)

        start = time.time()
        new_centroids, new_clusters = self.repeat_until_convergence(centroids, clusters, train_data)
        end = time.time()
        print("time for coveregence point ", end - start)

        start2 = time.time()
        labels_of_cluster = self.count_max_labels_of_cluster(new_clusters)
        end2 = time.time()
        print("time for labelling clusters ", end2 - start2)

        self.predict_digit(test_data, new_centroids, labels_of_cluster)

n = KNN()
k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
train_data, test_data = n.loadData()
for i in k:
    load_data = n.runKNN(train_data, test_data, i)
