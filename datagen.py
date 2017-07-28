#This script generates the entry list for training data to feed to our machine learning algorithms.

import pygrib as pg
import csv
import numpy as np
import datetime as dt

#To randomly generate non-dust dates, we'll use the mean and standard deviations of the latitudes and longitudes for all dust entries.
avglat = 32.4961
avglon = -107.6339
stdlat = 2.2511
stdlon = 2.4215

entries = open('events.csv',newline='')
csvr = csv.reader(entries,delimiter=',',quotechar='|')

outfile = open('entries.csv','w',newline='') #This is what the recorder script will read
csvw = csv.writer(outfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)

d1 = dt.date(2007,4,13) #start date of 13km RUC
d2 = dt.date(2012,4,30) #end date of 13km RUC

events = np.zeros(3) #create a numpy array with list of events
for row in csvr:
    vec = np.array([float(x) for x in row])
    events = np.vstack((events,vec))
events = events[1:] #chop off that row of zeros

grbs = pg.open('gribs/ruc2anl_130_20070413_1800_000.grb2') #get a reference for coordinates
lats,lons = grbs[1].latlons()
grbs.close()

x = d1
while x <= d2:
    xstr = str(x).replace('-','')
    date = float(xstr)
    if date in events:
        dateInd = np.where(events==date)
        for k in range(len(dateInd[0])):
            lat = events[dateInd[0][k]][0]
            lon = events[dateInd[0][k]][1]

            X = np.where(abs(lats-lat)<=0.01) #Get a list of candidates with close enough latitude
            indices = list(zip(*X)) #We have a list of indices of the candidates.
            candidates = [lons[i][j] - lon for (i,j) in indices] #get difference in longitude of each candidate
            index = np.abs(candidates).argmin() #index of closest longitude
            latIndex,lonIndex = indices[index] #get the indices for the latitude and longitude

            csvw.writerow(['gribs/ruc2anl_130_'+xstr+'_1800_000.grb2',latIndex,lonIndex,1]) #Write to the CSV an entry with the file name, indices, and that it is a dust event (1)
    else:
        lat = np.random.normal(avglat,stdlat)
        lon = np.random.normal(avglon,stdlon)
        X = np.where(abs(lats-lat)<=0.01) #Get a list of candidates with close enough latitude
        indices = list(zip(*X)) #We have a list of indices of the candidates.
        candidates = [lons[i][j] - lon for (i,j) in indices] #get difference in longitude of each candidate
        index = np.abs(candidates).argmin() #index of closest longitude
        latIndex,lonIndex = indices[index] #get the indices for the latitude and longitude
        csvw.writerow(['gribs/ruc2anl_130_'+xstr+'_1800_000.grb2',latIndex,lonIndex,0]) #Write to the CSV an entry with the file name, indices, and that it is a nondust event (0)
    x = x + dt.timedelta(days=1)


