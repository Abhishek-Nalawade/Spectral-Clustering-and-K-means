import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class queue:
    def __init__(self):
        self.pr_queue = list()

    def remove(self):
        a = self.pr_queue.pop()
        return a

    def add(self, new):
        self.pr_queue.insert(0, new)
        return

class Spectral_Clustering:
    def __init__(self, no_of_classes, data):
        self.data = data
        self.X = 1
        self.Y = 1
        self.centered_X = 1
        self.centered_Y = 1
        self.no_of_classes = no_of_classes
        self.queue1 = queue()
        self.classes = list()

    def load_data(self):
        data = loadmat(self.data)
        #data = loadmat('hw06-data2.mat')
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
        self.centered_X = np.reshape(self.centered_X, (1,self.X.shape[0]))
        self.centered_Y = np.reshape(self.centered_Y, (1,self.Y.shape[0]))
        self.queue1.add([self.centered_X, self.centered_Y])
        return

    def plot_initial_data(self):
        plt.scatter(self.X, self.Y)
        plt.show()
        plt.scatter(self.centered_X, self.centered_Y)
        plt.show()
        return

    def form_W_matrix(self):
        #print("shape ",self.centered_X.shape)
        coor = self.queue1.remove()
        self.centered_X = coor[0]
        self.centered_Y = coor[1]
        z = np.zeros((self.centered_X.shape[1],1))
        Wx = self.centered_X + z
        Wy = self.centered_Y + z
        #print("new ",Wx)
        #print(Wx - self.centered_X.T)
        W = ((Wx - self.centered_X.T)**2 + (Wy - self.centered_Y.T)**2)
        W = -(W/10)
        W = np.exp(W)
        return W

    def form_D_matrix(self, W):
        I = np.eye(W.shape[0])
        D = np.sum(W, axis = 1, keepdims = True)
        D = I * D
        return D

    def classify(self, w, d, colour):
        L = d - w
        #print("sum ",np.sum(L[500,:]))
        dinv = np.linalg.inv(d)
        mat = np.dot(dinv**(1/2), np.dot(L, dinv**(1/2)))
        eigval, eigvec = np.linalg.eig(mat)
        #print(np.argmin(eigval))
        sorteval = sorted(eigval)
        #print("shape ",eigvec.shape)
        ind = np.where(eigval == sorteval[1])
        y1 = eigvec[:,ind[0]]
        #y1 = np.dot(dinv**(1/2), y1)
        pos = np.where(y1 > 0)
        neg = np.where(y1 <= 0)
        #print(neg[0])
        #classes = np.concatenate()
        new_Xpos = np.reshape(self.centered_X[0,pos[0][:]], (1, self.centered_X[0,pos[0][:]].shape[0]))
        new_Ypos = np.reshape(self.centered_Y[0,pos[0][:]], (1, self.centered_Y[0,pos[0][:]].shape[0]))

        new_Xneg = np.reshape(self.centered_X[0,neg[0][:]], (1, self.centered_X[0,neg[0][:]].shape[0]))
        new_Yneg = np.reshape(self.centered_Y[0,neg[0][:]], (1, self.centered_Y[0,neg[0][:]].shape[0]))

        self.classes.append([[new_Xpos, new_Ypos],[new_Xneg, new_Yneg]])

        plt.scatter(new_Xpos, new_Ypos, color = colour[0])
        plt.scatter(new_Xneg, new_Yneg, color = colour[1])
        plt.pause(0.05)
        self.queue1.add([new_Xpos, new_Ypos])
        self.queue1.add([new_Xneg, new_Yneg])
        return

    def run_algo(self):
        self.load_data()
        self.center_data()
        color = [['green','gray'],['purple','thistle'],['cyan','slateblue'],['skyblue','violet'],['olive','teal'],['blue','red'],['orange','black'],['lime','tan']]
        for i in range(self.no_of_classes-1):
            w = self.form_W_matrix()
            d = self.form_D_matrix(w)
            self.classify(w, d, color[i])
        plt.show()
        return self.classes
