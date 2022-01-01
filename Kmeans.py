import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class K_means:
    def __init__(self, no_of_classes, data):
        self.data = data
        self.X = 1
        self.Y = 1
        self.centered_X = 1
        self.centered_Y = 1
        self.no_of_classes = no_of_classes

    def load_data(self):
        data = loadmat(self.data)
        #print(data)
        x = data['X']
        self.X = x[0,:]
        self.Y = x[1,:]
        return

    def center_data(self):
        meanX = np.sum(self.X)/self.X.shape[0]
        meanY = np.sum(self.Y)/self.Y.shape[0]
        self.centered_X = self.X - meanX
        self.centered_Y = self.Y - meanY
        return

    def plot_initial_data(self):
        plt.scatter(self.X, self.Y)
        plt.show()
        plt.scatter(self.centered_X, self.centered_Y)
        plt.show()
        return

    def initialize_Us(self):
        rand_int = np.random.choice(self.X.shape[0], self.no_of_classes, replace = False)
        Uxs = self.centered_X[rand_int]
        Uys = self.centered_Y[rand_int]
        Uxs = np.reshape(Uxs, (self.no_of_classes, 1))
        Uys = np.reshape(Uys, (self.no_of_classes, 1))
        centroids = np.concatenate((Uxs, Uys), axis = 1)
        return centroids

    def assign_labels(self, centroids):
        all_distances = 10 * np.ones((self.X.shape[0],1))
        for i in range(self.no_of_classes):
            distX = ((self.centered_X) - (centroids[i,0]))**2
            distY = ((self.centered_Y) - (centroids[i,1]))**2
            dist = (distX + distY)**(1/2)
            dist = np.reshape(dist, (dist.shape[0],1))
            if(i == 0):
                all_distances = dist
            else:
                all_distances = np.concatenate((all_distances, dist), axis = 1)

        classes = np.argmin(all_distances, axis = 1)
        return classes

    def recomputeU(self, classes):
        Us = list()
        for i in range(self.no_of_classes):
            loc = np.where(classes == i)
            loc = np.array(loc)
            Ux = np.sum(self.centered_X[loc[0]])/len(loc[0])
            Uy = np.sum(self.centered_Y[loc[0]])/len(loc[0])
            Us.append([Ux,Uy])
        Us = np.array(Us)
        return Us

    def plot_new(self, classes):
        mark = ['red', 'orange', 'blue', 'green', 'olive', 'black', 'cyan', 'brown']
        for i in range(self.no_of_classes):
            loc = np.where(classes == i)
            loc = np.array(loc)
            plt.scatter(self.centered_X[loc[0,:]], self.centered_Y[loc[0,:]], color = mark[i])
            plt.pause(0.05)
        #plt.show()
        return

    def compute_L2norm(self, newUs, prevUs):
        diff_sq = (prevUs - newUs)**2
        L2norm = np.sum(diff_sq, axis = 1)**(1/2)
        metric = np.sum(L2norm)
        print("distance is ",metric)
        return metric

    def run_algo(self):
        self.load_data()
        self.center_data()
        classes = 10 * np.ones((self.X.shape[0],1))
        prevUs = np.random.rand(self.no_of_classes, 2)
        metric = 10
        i = 0
        while(metric > 0.01):
            if i == 0:
                Us = self.initialize_Us()
            classes = self.assign_labels(Us)
            Us = self.recomputeU(classes)
            metric = self.compute_L2norm(Us, prevUs)
            prevUs = Us
            i += 1
            self.plot_new(classes)
        print("The algorithm has converged")
        plt.show()
        return classes
