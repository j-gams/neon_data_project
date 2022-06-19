print ("initializing (importing packages)")
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import earthpy as et
import earthpy.plot as ep
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import shapefile
from pyproj import Transformer

#conda activate code_match; cd work/earthlab/munge_data/; python analyze_clipped.py
#stolen helper

def inv_index(xhat, yhat, ox, oy, pw, ph):
    ph = abs(ph)
    j = (oy - (ph * yhat)) #+ 0.00001
    i = (ox + (pw * xhat)) #+ 0.00001
    return i, j

def get_index(x, y, ox, oy, pw, ph):
    ph = abs(ph)
    j = math.floor((oy-y)/ph)
    i = math.floor((x-ox)/pw)
    return i, j

filters = [np.array([[1,  1, 1.0],
                     [0,  0, 0.0],
                     [-1,-1,-1.0]]),
           np.array([[0,   1,  1.4,],
                     [-1,  0,  1],
                     [-1.4,-1, 0]]),
           np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]]),
           np.array([[-1.4, -1, 0],
                     [-1, 0, 1],
                     [0, 1, 1.4]])
           ]

print ("done importing packages")

#load files
print("loading nlcd data")
nlcd = rxr.open_rasterio("nlcd_raw/clipped_nlcd.tif", masked=True).squeeze()
print("loacing srtm data")
srtm = rxr.open_rasterio("srtm_raw/combined.tif", masked=True).squeeze()

#"""    ###     ###     ###     ###
srtm_g = gdal.Open("srtm_raw/combined.tif", gdal.GA_ReadOnly)
print(type(srtm_g))
ox, pw, xskew, oy, yskew, ph = srtm_g.GetGeoTransform()
nd_value = 100000000
arr = srtm_g.ReadAsArray()
#print(arr)
#del srtm_g
#window_size = (3,3)
#padding_y = (2,2)
#padding_x = (2,2)
#padded_arr = np.pad(arr, pad_width=(padding_y, padding_x), mode='constant', constant_values = nd_value)
#print(nd_value)
#print(padded_arr)
#print(padded_arr[0, 0])
#lyr = srtm_g.GetLayer()
#print(type(lyr))
#coords = [(feat.geometry().GetX(), feat.geometry.GetY()) for feat in lyr]
#coords = np.array(coords)
#x = coords.T[0]
#y = coords.T[1]
del srtm_g#, lyr
#"""

print("loading gedi data")
#gedi_pts = gpd.read_file("GEDI_2B_clean/GEDI_2B_clean.shp")
gedi_pts = shapefile.Reader("GEDI_2B_clean/GEDI_2B_clean.shp")
#print(type(gedi_pts))
print("done")
#print(gedi_pts.shapeTypeName)
print(len(gedi_pts), "points")
for f in range(len(gedi_pts.fields)):
    print(f, gedi_pts.fields[f])
grecs = gedi_pts.shapeRecords()
#print(dir(grecs[3]))
#for name in dir(grecs[3]):
#    print(name)
#print("...")

#print(type(grecs[1].shape))
#print(grecs[1].shape)
#print("...")
#print(grecs[0].record)

cut = len(gedi_pts)
print("running on first", cut, "points")
nlcd_trans = Transformer.from_crs("EPSG:4326", nlcd.rio.crs, always_xy=True)
nlcd_vals = []
srtm_elev = []
print("getting nlcd values")
print(arr.shape)
aspectinfo = np.zeros((arr.shape[0], arr.shape[1], len(filters)))
for i in range(cut): #len(gedi_pts)):
    a, b = [grecs[i].record[2], grecs[i].record[3]]
    print(a, b)
    #print()
    break
    xp, yp = nlcd_trans.transform(b, a)
    nlcd_vals.append(float(nlcd.sel(x=xp,y=yp, method='nearest').values))
    srtm_elev.append(float(srtm.sel(x=xp,y=yp, method='nearest').values))

    ##get coords, then neighbors coords
    #height. width?
    if False:
        ix, jy = get_index(xp, yp, ox, oy, pw, ph)
        fresults = []
        for f in filters:
            fresults.append(0)
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    nx = ix + j
                    ny = jy + k
                    if j == 0 and k == 0:
                        #check
                        if not nlcd_vals[-1] == float(nlcd.sel(x=inv_index(ix,jy,ox,oy,pw,ph)[0], y=inv_index(ix,jy,ox,oy,pw,ph)[1], method='nearest').values):
                            print("uhoh", xp, yp, " -> ", inv_index(ix, jy, ox, oy, pw, ph))
                    if nx >= 0 and nx < arr.shape[0] and ny >= 0 and ny < arr.shape[1]:
                        fresults[-1] += f[j+1, k+1]*arr[nx, ny]
        for ff in range(len(fresults)):
            aspectinfo[ix, jy, ff] = fresults[ff]
    #if str(nlcd_vals[-1]) != "nan":
    #    print(nlcd_vals[-1])
    if i % 50000 == 0:
        print(i)
#print(latlon)
print("making graphs")
print(nlcd_vals[:20])

plt.figure()
plt.hist(nlcd_vals)
plt.title("NLCD - land covers at GEDI data points")
plt.savefig("figures/gedi_distributions/land_cover_dist.png")
plt.show()
plt.cla()
plt.close()

plt.figure()
plt.hist(srtm_elev)
plt.title("SRTM - elevation at GEDI data points")
plt.savefig("figures/gedi_distributions/elevation_dist.png")
plt.show()
plt.close()
print("all done")

"""
nlcd_val_idx = {}
nlcd_graph = []
nlcd_freqs = []
for i in range(cut):#len(nlcd_vals)):
    print(type(nlcd_vals[i]))
    if nlcd_vals[i] in nlcd_graph:
        nlcd_freqs[nlcd_val_idx[nlcd_vals[i]]] += 1
    else:
        nlcd_val_idx[nlcd_vals[i]] = len(nlcd_graph)
        nlcd_graph.append(nlcd_vals[i])
        nlcd_freqs.append(1)
#do graph
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(nlcd_graph, nlcd_freqs)
plt.show()
"""
