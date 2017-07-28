import pygrib as pg
import numpy as np
def ingest(filename, x, y, selector):
    grbs = pg.open(filename)
    arr = np.array([grbs[sel].values[x][y] for sel in selector]) #list comprehension
    #arr = map(lambda grb: grb.values[x][y],grbs) #map (lazy evaluation!)
    #for grb in grbs:
    #    arr.append(grb.values[x][y])
    arr = np.concatenate([barr,parr])
    return arr
#grbname = 'ruc2.t19z.bgrb13anl.grib2'
#vec = ingest(grbname,0,0)
#print(vec)
