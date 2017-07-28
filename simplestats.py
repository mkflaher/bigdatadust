import numpy as np
import csv
with open('dustpc.csv',newline='') as dust:
    pc0 = np.array([])
    pc1 = np.array([])
    pc2 = np.array([])
    pc3 = np.array([])
    pc4 = np.array([])

    csvr = csv.reader(dust,delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
    for row in csvr:
        pc0 = np.append(pc0,row[0])
        pc1 = np.append(pc1,row[1])
        pc2 = np.append(pc2,row[2])
        pc3 = np.append(pc3,row[3])
        pc4 = np.append(pc4,row[4])

    pc0 = np.array([float(x) for x in pc0])
    pc1 = np.array([float(x) for x in pc1])
    pc2 = np.array([float(x) for x in pc2])
    pc3 = np.array([float(x) for x in pc3])
    pc4 = np.array([float(x) for x in pc4])
    print(pc0,pc1,pc2,pc3,pc4)
    print('dust stats\n')
    print('PC1\n',np.mean(pc0),np.std(pc0))
    print('PC2\n',np.mean(pc1),np.std(pc1))
    print('PC3\n',np.mean(pc2),np.std(pc2))
    print('PC4\n',np.mean(pc3),np.std(pc3))
    print('PC5\n',np.mean(pc4),np.std(pc4))

with open('ndustpc.csv',newline='') as ndust:
    pc0 = np.array([])
    pc1 = np.array([])
    pc2 = np.array([])
    pc3 = np.array([])
    pc4 = np.array([])

    csvr = csv.reader(ndust,delimiter=',',quotechar='"',quoting=csv.QUOTE_NONE)
    for row in csvr:
        pc0 = np.append(pc0,row[0])
        pc1 = np.append(pc1,row[1])
        pc2 = np.append(pc2,row[2])
        pc3 = np.append(pc3,row[3])
        pc4 = np.append(pc4,row[4])

    pc0 = np.array([float(x) for x in pc0])
    pc1 = np.array([float(x) for x in pc1])
    pc2 = np.array([float(x) for x in pc2])
    pc3 = np.array([float(x) for x in pc3])
    pc4 = np.array([float(x) for x in pc4])
    print('nondust stats\n')
    print('PC1\n',np.mean(pc0),np.std(pc0))
    print('PC2\n',np.mean(pc1),np.std(pc1))
    print('PC3\n',np.mean(pc2),np.std(pc2))
    print('PC4\n',np.mean(pc3),np.std(pc3))
    print('PC5\n',np.mean(pc4),np.std(pc4))
