#This script takes the csv obtained from the RUC/RAP data, and outputs a csv of its prinicpal components. The principal components will then be used for further data processing.

import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
dust_matrix = np.array([])
nondust_matrix = np.array([])
dimensions = 10 
veclength = 0 #define a global variable for vector length
with open('traindata.csv',newline='') as traindata:
    csvr = csv.reader(traindata,delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
    veclength = len(next(csvr)) - 1
    print(veclength)
    traindata.seek(0)
    matrix = np.zeros(veclength+1) #sets up a row of padding to vstack
    for row in csvr:
        rowvec = [float(x) for x in row]
        matrix = np.vstack((matrix,rowvec))
matrix = matrix[1:] #chop off that zeros row
classes = matrix[:,0] #0 or 1 for nondust or dust
matrix = matrix[:,1:] #get rid of the first column for PCA
#print(len(dust_matrix))
#print(len(nondust_matrix))

matrix = matrix.T #transpose it for scatter matrix calculations
mean_vec = np.array([])
#print(matrix)
#print(np.shape(matrix))
for row in matrix: #transpose it to get the average of every column
    mean_vec=np.append(mean_vec,np.mean(row))
mean_vec = mean_vec.reshape(-1,1) #whoever wrote that stupid pca tutorial did too many transposes
#print('Mean vector: \n', mean_vec)

#scatter matrix
scat_mat = np.zeros((veclength,veclength))
for i in range(len(matrix)):
    scat_mat += (matrix[:,i].reshape(-1,1) -  mean_vec).dot((matrix[:,i].reshape(-1,1) - mean_vec).reshape(1,-1))

eig_val, eig_vec = np.linalg.eig(scat_mat) #get the eigenvectors and values
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))] #list of val,vec tuples
eig_pairs.sort(key=lambda x: x[0], reverse=True) #sort by val in desc order
#need to get the first k eigenvectors from this list.
mat_w = eig_pairs[0][1].reshape(-1,1)
for vec in range(1,dimensions):
    mat_w = np.hstack((mat_w, eig_pairs[vec][1].reshape(-1,1)))

transformed = mat_w.T.dot(matrix) #Gives us the data in the first k principle components
average = np.ones(np.shape(transformed)) * np.mean(transformed) #mean of values in the transformed matrix
transformed = (transformed - average)/np.std(transformed) #gives us normalized principal components
transformed = transformed.T
print(np.shape(transformed))
dusttf = np.zeros(dimensions)
ndusttf = np.zeros(dimensions)
for i in range(len(transformed)):
    if classes[i]==1:
        dusttf = np.vstack((dusttf,transformed[i]))
    else:
        ndusttf = np.vstack((ndusttf,transformed[i]))
dusttf = dusttf[1:]
ndusttf = ndusttf[1:]
for i in range(dimensions):
    for j in range(dimensions):
        if j>i:
            font = {'family' : 'normal',
                    'weight' : 'normal',
                    'size' : 22}
            matplotlib.rc('font',**font)
                    


            plt.plot(dusttf[:,i],dusttf[:,j],'o',color='blue',label='Dust Event')
            plt.plot(ndusttf[:,i],ndusttf[:,j],'^',color='red',alpha=0.1,label='Non-Dust Event')
            plt.title('Principal Component %d vs %d' % (i,j))
            plt.xlabel(i)
            plt.ylabel(j)
            plt.legend()
            plt.show()

#dusttf = transformed.T[0:dustlen]
#ndusttf = transformed.T[dustlen:-1]
##print('dust\n',dusttf)
##print('ndust\n',ndusttf)
#print('transformed\n',transformed.T)
#with open('dustpc.csv','w',newline='') as dustpc:
#    csvw = csv.writer(dustpc, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for entry in dusttf:
#        row = [float(np.real(x)) for x in entry]
#        csvw.writerow(row)
classes = classes.reshape(-1,1)
transformed = np.hstack((classes,transformed)) #reunite the entries with their classifications
print(transformed)
with open('trainpc.csv','w',newline='') as trainpc:
    csvw = csv.writer(trainpc, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for entry in transformed:
        row = [float(np.real(x)) for x in entry]
        csvw.writerow(row)
