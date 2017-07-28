import ingestor as ing
import pygrib as pg
import numpy as np
import csv
#This script uses ingestor to get data we're interested in into a CSV file for later, easier processing
#We want to open a CSV file that tells us which data to read, then crunch it into an output file with all of the data from each entry
#each row of the input CSV should be as follows: file, x, y
inputArr = []
with open('entries.csv',newline='') as entries:
    csvr = csv.reader(entries, delimiter=',', quotechar="|")
    for row in csvr:
        inputArr.append((row[0],row[1],row[2],row[3]))
#Now that we have a list of points to collect, we gather the file data and put it into a csv
with open('traindata.csv','w',newline='') as ruc:
    csvw = csv.writer(ruc, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
    for entry in inputArr:
        vector = ing.ingest(entry[0],int(entry[1]),int(entry[2]))
        if not np.all(vector==0):
            csvw.writerow(np.insert(vector,0,entry[3]))

