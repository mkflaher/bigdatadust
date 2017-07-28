#This script does not use a selector to take out particular features of the dataset; it is entirely raw (and big!)
import pygrib as pg
import numpy as np
def ingest(filename, x, y):
    try:
        grbs = pg.open(filename)
        arr = np.array([grb.values[x][y] for grb in grbs]) #list comprehension
        #arr = map(lambda grb: grb.values[x][y],grbs) #map (lazy evaluation!)
        #for grb in grbs:
        #    arr.append(grb.values[x][y])
        grbs.close()
        return arr
    except OSError:
        print(filename+' not found.')
        return np.array(0)
#grbname = 'ruc2.t19z.bgrb13anl.grib2'
#vec = ingest(grbname,0,0)
#print(vec)
