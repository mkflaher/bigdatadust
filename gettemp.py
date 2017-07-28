import pygrib as pg
import numpy as np

filename="gribs/ruc1320110418/ruc2.t22z.pgrb13anl.grib2"
grbs = pg.open(filename)

for grb in grbs:
    if grb.parameterName=="Temperature" and grb.level==2:
        print(grb.values)
        print(grb.latlons())
