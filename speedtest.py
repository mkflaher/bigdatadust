import ingestor as ing
import csv
selector = [1,2,3,4,5]
inputArr = []
with open('entries.csv',newline='') as entries:
    csvr = csv.reader(entries, delimiter=',', quotechar="|")
    for row in csvr:
        inputArr.append((row[0],row[1],row[2]))
for entry in inputArr:
    vec = ing.ingest(entry[0],int(entry[1]),int(entry[2]),selector)
    print(entry)
