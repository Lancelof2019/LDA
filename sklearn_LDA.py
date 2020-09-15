import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
from copy import deepcopy
from matplotlib import pyplot as plt
import seaborn as sns
iris=datasets.load_iris()

#print(iris)
X=iris.data[:,2:4]
print(X)
y=iris.target
print(y)
print('--------------------------------------------------')

LDA=LinearDiscriminantAnalysis(n_components=2)
LDA.fit(iris.data,iris.target)
#训练出来的模型
proj_LDA=LDA.transform(iris.data)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(proj_LDA[:,0],proj_LDA[:,1])
plt.show()

#print(proj_LDA)
#check_point=list(set(iris.target))

#print(check_point)
#i=0
'''for i in check_point:
    #LDA.fit(iris.data,iris.target)
    #plt.scatter(proj_LDA[i])
    print(proj_LDA[i])
#plt.show()
'''



sns.scatterplot(proj_LDA[:, 0], proj_LDA[:, 1], hue=iris.target)

plt.show()
