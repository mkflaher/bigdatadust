import selectiveIngestor as ing
import pygrib as pg
import csv
#This script uses ingestor to get data we're interested in into a CSV file for later, easier processing
#We want to open a CSV file that tells us which data to read, then crunch it into an output file with all of the data from each entry
#each row of the input CSV should be as follows: file, x, y
selector=[1,2,3,4,5] #This will determine which variables to select from each RUC file
inputArr = []
with open('dustentries.csv',newline='') as entries:
    csvr = csv.reader(entries, delimiter=',', quotechar="|")
    for row in csvr:
        inputArr.append((row[0],row[1],row[2]))
#Now that we have a list of points to collect, we gather the file data and put it into a csv
with open('dustdata.csv','w',newline='') as ruc:
    csvw = csv.writer(ruc, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for entry in inputArr:
        vector = ing.ingest(entry[0],int(entry[1]),int(entry[2]),selector)
        csvw.writerow(vector)
with open('nondustentries.csv',newline='') as entries:
    csvr = csv.reader(entries, delimiter=',', quotechar="|")
    for row in csvr:
        inputArr.append((row[0],row[1],row[2]))
#Now that we have a list of points to collect, we gather the file data and put it into a csv
with open('nondustdata.csv','w',newline='') as ruc:
    csvw = csv.writer(ruc, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for entry in inputArr:
        vector = ing.ingest(entry[0],int(entry[1]),int(entry[2]),selector)
        csvw.writerow(vector)

