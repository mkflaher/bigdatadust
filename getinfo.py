import numpy as np
import pygrib as pg

filename = "gribs/ruc1320110418/ruc2.t22z.pgrb13anl.grib2" #Remember to put something here!
grbs = pg.open(filename)

#for grb in grbs:
#    print(grb.parameterName)
#   print(grb.values.shape)


lats, lons = grbs[1].latlons()
print(lats.min(), lats.max())
print(lons.min(), lons.max())
