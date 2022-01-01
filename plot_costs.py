import matplotlib.pyplot as plt
from Cost import *


classes = [2, 4, 8]
kmeans_costs = list()
spc_costs = list()
for no_classes in classes:
    data = 'hw06-data1.mat'
    # data = 'hw06-data2.mat'
    cost = Cost(no_classes, data)
    kcost, spcost = cost.get_cost()

    kmeans_costs.append(kcost)
    spc_costs.append(spcost)


plt.title("K-means costs")
plt.plot(classes, kmeans_costs, 'bo')
for i in range(len(kmeans_costs)):
    plt.text(classes[i], kmeans_costs[i], str(kmeans_costs[i]))
plt.plot(classes, kmeans_costs)
plt.show()


plt.title("Spectral Clustering costs")
plt.plot(classes, spc_costs, 'bo')
for i in range(len(spc_costs)):
    plt.text(classes[i], spc_costs[i], str(spc_costs[i]))
plt.plot(classes, spc_costs)
plt.show()
