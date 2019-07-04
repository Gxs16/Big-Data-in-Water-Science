from netCDF4 import Dataset
import numpy as np
from sklearn.decomposition import PCA
import sklearn.model_selection as sk_model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def inverse(Judge, W):
        point = np.empty([10988,3])
        s = 0
        for i in range(89):
                for j in range(180):
                        if Judge[i,j] == 1:
                                point[s,:] = (88-2*i,2*j,W[s])
                                s+=1
        return point

Info = Dataset('sst.mnmean.nc') #Read .nc file
SST = Info.variables['sst'][:, :, :].data #Extract the data of temperature
del Info
location = ((134, 44), 
       (94, 44), 
       (57, 37), 
       (39, 44), 
       (21, 36), 
       (9, 16), 
       (172, 59), 
       (158, 28), 
       (65, 64), 
       (0, 10)) #Certificate the indexes of the target locations 

A = SST > SST.min()
Judge = A[0,:,:]#Compute a True-false matrix if the area is sea, Judge = True
del A
new_SST = np.zeros((len(SST), int(Judge.sum())))  
s = 0
for i in range(89):
    for j in range(180):
        if Judge[i, j] == 1:
            new_SST[:, s] += SST[:, i, j]
            s += 1  # Reshape SST from 3D into 2D
del SST
new_SST = new_SST[-1202:]#Extract the data from 1919-1 to 2019-2

X_training = new_SST[0:960]#Extract the data from 1919-1 to 1999-12 as training data 

month_mean = np.zeros((12, int(np.sum(Judge))))
for i in range(12):
    month_mean[i] = np.copy(np.average(X_training[i: :12, :], axis = 0)) #Compute the month average temperature in training data
for i in range(12):
    new_SST[i: :12, :] -= month_mean[i]#All data minus the average temperature

X_training = new_SST[0:960]#Extract the data from 1919-1 to 1999-12 as training data
X_test = new_SST[960:]#Extract the data from 1999-1 to 2019-12 as test data
loc = np.array([1511, 1925, 3160, 4012, 4174, 5164, 5215, 5255, 7401, 8129])#The index in new_SST of ten target points
Y = np.empty([1201,10])
s = 0
for i in loc:
        Y[:,s] = new_SST[1: ,i]
        s += 1
Y_training = Y[:960]
Y_test = Y[960:]#Extract output data 

pca = PCA(n_components=100)

X_mean = np.average(X_training, axis = 0)
X_training -= X_mean
X_test -= X_mean
pca.fit(X_training)

X_training_new = pca.transform(X_training)
X_test_new = pca.transform(X_test)
Y_fore = np.empty(np.shape(Y_test))
Y_output = np.empty([10])
alpha = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 100, 1000])
alpha2 = np.empty([10])
for i in range(10):
        Output = np.empty([8])
        for j in range(8):
                reg = linear_model.Lasso(alpha=alpha[j], max_iter=10000)
                Output[j] = abs(np.mean(sk_model_selection.cross_val_score(reg, X_training_new, Y_training[:,i].T, scoring='neg_mean_squared_error', cv = 10)))**0.5
        alpha2[i] = alpha[np.where(Output == np.min(Output))]
lonlat = ('68N,0','56N,18E','32N,44W','16N,42E','14N,114E','0,78E','0,92W','0,172W','30S,16W','40S,130E')
NSE = np.empty([10])
for i in range(10):
        reg = linear_model.Lasso(alpha=alpha2[i], max_iter=10000)
        reg.fit(X_training_new, Y_training[:,i])
        Y_fore[:,i] = np.sum(X_test_new[:-1] * reg.coef_, axis=1) + reg.intercept_  # 注意此时y是距平后的y
        Y_output[i] = np.sum(X_test_new[-1] * reg.coef_) + reg.intercept_ + month_mean[2, loc[i]]
        NSE[i] = 1 - np.sum((Y_fore[:,i] - Y_test[:,i]) ** 2) / np.sum((Y_test[:,i] - np.average(Y_test[:,i])) ** 2)  # 还原y后做NSE
        for j in range(12):
                Y_fore[j: :12, i] += month_mean[j, loc[i]]
                Y_test[j: :12, i] += month_mean[j, loc[i]]
        fig=plt.figure(lonlat[i],figsize=(8,5),dpi=300)
        plt.plot(Y_fore[:,i],color='red',linewidth=0.4)
        plt.plot(Y_test[:,i],color='blue',linewidth=0.4)
        plt.title(lonlat[i])
        plt.xlabel('Month')
        plt.ylabel('Celsius degree')
        plt.legend(('Forecast','True'))
        plt.savefig(lonlat[i]+'.png')
np.savetxt('Y_fore.csv',Y_fore, delimiter=',')
np.savetxt('Predication.csv',Y_output, delimiter=',')
np.savetxt('Hypercoeffecient.csv',alpha2, delimiter=',')
np.savetxt('W', pca.components_, delimiter=',')
np.savetxt('NSE',NSE, delimiter=',')