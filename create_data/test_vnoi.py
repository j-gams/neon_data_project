### Written by Jerry Gammie @j-gams

print("importing...")
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import earthpy as et
import earthpy.plot as ep
from osgeo import gdal
from osgeo import ogr
import shapefile
from longsgis import voronoiDiagram4plg
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from geovoronoi import voronoi_regions_from_coords, points_to_coords

#conda activate code_match; cd work/earthlab/munge_data/; python test_vnoi.py

def idx_pixctr(ix, iy, ulh, ulv, psh, psv, mode='ul'):
    offsetx = 0
    offsety = 0
    if mode=='ctr':
        offsetx = psh/2
        offsety = psv/2
    cx = ulh + (ix * psh) + offsetx
    cy = ulv - (iy * psv) + offsety
    return cx, cy

ecos_g = gdal.Open("../raw_data/ecos_wue/WUE_Median_Composite_AOI.tif")
print("loading ecostress data...")
#ecos_rband = ecos_g.GetRasterBand(1)
ecos_ndv = -9999#ecos_rband.GetNoDataValue()
ecos_rsize = (ecos_g.RasterXSize, ecos_g.RasterYSize)
UL_h, h_spac, _, UL_v, _, v_spac = ecos_g.GetGeoTransform()
#account for negative pixel size:
v_spac = abs(v_spac)
print("ECOSTRESS crs info:", UL_h, UL_v, h_spac, v_spac)
print("ECOSTRESS raster size:", ecos_rsize)
print("ECOSTRESS no data value:", ecos_ndv)
ecos_npar = ecos_g.ReadAsArray().transpose()
print(ecos_npar.shape)

crs = {'init': "epsg:4326"}
del ecos_g
#gedi_bup = gpd.read_file("GEDI_2B_clean/GEDI_2B_clean.shp")
print("loading points in geopandas")
# need to load with geopandas...
print("working by pixel...")
for i in range(ecos_rsize[0]):
     for j in range(ecos_rsize[1]):
         if ecos_npar[i, j] != ecos_ndv:
             ##read in points within bounding box...?
             print("loading points subject to bound")
             midlon, midlat = idx_pixctr(i, j, UL_h, UL_v, h_spac, v_spac)
             bblats = [-(4 * v_spac) + midlat, (5 * v_spac) + midlat, (5 * v_spac) + midlat, -(4 * v_spac) + midlat]
             bblons = [(4*h_spac) + midlon, (4*h_spac) + midlon, (5*h_spac) + midlon, (5*h_spac) + midlon]

             tbbox = Polygon(zip(bblons, bblats))
             pixbbox = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[tbbox])
             print(pixbbox.geometry)

             gedi_bup = gpd.read_file("../raw_data/gedi_pts/GEDI_2B_clean.shp", bbox = tbbox)
             if gedi_bup.shape[0] > 1:
                 print(gedi_bup.geometry[0].x, gedi_bup.geometry[0].y)
             else:
                 print("got nothin")
                 continue
             print("making voronoi diagram")

             fig, ax = plt.subplots(figsize=(12, 10))
             pixbbox.plot(ax=ax, color="gray")
             gedi_bup.plot(ax=ax, markersize=3.5,color="black")
             plt.show()
             print("CRS", gedi_bup.crs)
             pixbbox=pixbbox.to_crs(epsg=4326)
             gdf_proj = gedi_bup.to_crs(pixbbox.crs)

             pbb_shape = cascaded_union(pixbbox.geometry)
             """"""
             gdf_coords = points_to_coords(gdf_proj.geometry)
             poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(gdf_coords, pbb_shape)
             #try plotting
             fig, ax = subplot_for_map()
             plot_voronoi_polys_with_points_in_area(ax, boundary_shapes, poly_shapes, pts, poly_to_pt_assignments)
             ax.set_title("voronoi")
             plt.tight_layout()
             plt.show()

             #pvd = voronoiDiagram4plg(gedi_bup, pixpoly)
             #pvd.plot()
             """bdry_wrap = shapefile.Writer("tpath")
             bdry_wrap.field('X', 'F', 10, 5)
             bdry_wrap.field('Y', 'F', 10,5)
             bdry_wrap.poly([[UL_h, UL_v],
                             [UL_h + ecos_h_spac, UL_v],
                             [UL_h + ecos_h_spac, UL_v + ecos_v_spac],
                             [UL_h, UL_v + ecos_v_spac]])"""

