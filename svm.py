import numpy as np
from sklearn import svm
import csv

#How this script should work:
#Two arrays of vectors: One for a list of vectors, the other for which class they belong to.

arr = np.zeros(5)
classes = np.array([])
with open('ndustpc.csv',newline='') as nondust:
    csvr = csv.reader(nondust, delimiter=',', quotechar='|')
    for row in csvr:
        arr = np.vstack((arr,row))
        classes=np.append(classes,0) #0 identifies non-dust days.
with open('dustpc.csv',newline='') as dust:
    csvr = csv.reader(dust, delimiter=',', quotechar='|')
    for row in csvr:
        arr = np.vstack((arr,row))
        classes=np.append(classes,1) #1 identifies dust days.

#Now that we have data formatted as we need, we can start with the SVM.
clf = svm.SVC()
clf.fit(arr,classes)

#Now if we want to predict an entry, we can do it here.
