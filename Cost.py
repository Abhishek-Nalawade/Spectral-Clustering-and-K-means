from Kmeans import *
from SpectralClustering import *

class Cost:
    def __init__(self, no_classes, data):
        self.no_of_classes = no_classes
        self.kmeans = K_means(self.no_of_classes, data)
        self.spc = Spectral_Clustering(self.no_of_classes, data)
        self.X = 1
        self.Y = 1

    def run_algos(self):
        k_classes = self.kmeans.run_algo()
        spc_classes = self.spc.run_algo()
        self.X = self.kmeans.centered_X
        self.Y = self.kmeans.centered_Y
        self.X = np.reshape(self.X, (1, self.X.shape[0]))
        self.Y = np.reshape(self.Y, (1, self.Y.shape[0]))
        return k_classes, spc_classes

    def compute_kmeans_cost(self, classes):
        J = 0
        for i in range(self.no_of_classes):
            loc = np.where(classes == i)
            #print(self.X.shape)
            x = self.X[0,loc[0][:]]
            y = self.Y[0,loc[0][:]]
            x = np.reshape(x, (1, x.shape[0]))
            y = np.reshape(y, (1, y.shape[0]))
            z = np.zeros((x.shape[1],1))
            fX = x + z
            fY = x + y
            fX = fX - x.T
            fY = fY - y.T
            J1 = fX**2 + fY**2
            J1 = np.sum(J1, axis = 1, keepdims = True)
            J1 = (1/x.shape[1]) * np.sum(J1, axis = 0)
            J = J + J1
        J = (1/2) * J
        print("Cost for K-means: ",J[0])
        return J[0]

    def compute_spc_cost(self, classes):
        print(len(classes))
        J = 0
        for i in range(1, int(self.no_of_classes/2)+1):
            for j in range(2):
                #print(classes[-i][j][0].shape)
                x = classes[-i][j][0]
                y = classes[-i][j][1]

                z = np.zeros((x.shape[1],1))
                fX = x + z
                fY = x + y
                fX = fX - x.T
                fY = fY - y.T
                J1 = fX**2 + fY**2
                J1 = np.sum(J1, axis = 1, keepdims = True)
                J1 = (1/x.shape[1]) * np.sum(J1, axis = 0)
                J = J + J1
        J = (1/2) * J
        print("Cost for Spectral Clustering: ", J[0])
        return J[0]

    def get_cost(self):
        k_classes, spc_classes = self.run_algos()
        kcost = self.compute_kmeans_cost(k_classes)
        spcost = self.compute_spc_cost(spc_classes)
        return kcost, spcost
