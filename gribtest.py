import pygrib as pg
filename = 'gribs/rap_130_20120513_1800_000.grb2'
grbs = pg.open(filename)
for grb in grbs:
    print(grb)
