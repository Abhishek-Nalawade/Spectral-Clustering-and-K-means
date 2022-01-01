from Cost import Cost

no_classes = input("Enter the number of classes: ")
no_classes = int(no_classes)
data = 'hw06-data1.mat'
# data = 'hw06-data2.mat'
cost = Cost(no_classes, data)
kcost, spcost = cost.get_cost()
