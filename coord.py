#This script is just if we want the lat/lon of a coordinate for creating our dataset.
import pygrib as pg
import numpy as np

lat = float(input('latitude '))
lon = float(input('longitude '))

grbs = pg.open('gribs/ruc2anl_130_20070413_1800_000.grb2')
lats,lons = grbs[1].latlons()

x = np.where(abs(lats-lat)<=0.01) #Get a list of candidates with close enough latitude
indices = list(zip(*x)) #We have a list of indices of the candidates.
candidates = [lons[i][j] - lon for (i,j) in indices] #get difference in longitude of each candidate
index = np.abs(candidates).argmin() #index of closest longitude
latIndex,lonIndex = indices[index] #get the indices for the latitude and longitude
print(latIndex,lonIndex,lats[latIndex,lonIndex],lons[latIndex,lonIndex])

