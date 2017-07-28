#This is a simple program to read data from a GRIB file.
import numpy as np
import pygrib as pg
#import matplotlib.pyplot as plt #Leave commented if not running in jupyter

filename = "4-12-2012-1800-000.grb2" #Remember to put something here!
grbs = pg.open(filename) #Makes a file iterator

plaintxt = open("gribtext.txt", "w")
for grb in grbs: #Reads every gribmessage's toString.
    plaintxt.write(str(grb)+"\n")
    print(grb)
#Once a GRIB file from one of the RUC disks is read using this, I will determine how to proceed.
plaintxt.close()
grbs.close()

