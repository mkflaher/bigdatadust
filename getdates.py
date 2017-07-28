import datetime as dt
d1 = dt.date(2006,1,1)
d2 = dt.date(2012,2,28)

datelist = []
x = d1
while x != d2:
    datelist.append(x)
    x = x + dt.timedelta(days=1)
nondustdates = open('nondustentries.csv','w')
for date in datelist:
    nondustdates.write('gribs/ruc2anl_130_'+str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)+'_1800_000.grb2,121,125\n')
nondustdates.close()
