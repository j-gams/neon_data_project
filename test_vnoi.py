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

ecos_g = gdal.Open("ecostress_WUE/WUE_Median_Composite_AOI.tif")
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
             bblats = [- (4 * v_spac), (5 * v_spac), (5 * v_spac), - (4 * v_spac)]
             bblons = [(4*h_spac), (4*h_spac), (5*h_spac), (5*h_spac)]
             #bblats = [UL_v - (4*v_spac), UL_v + (5*v_spac), UL_v + (5*v_spac), UL_v - (4*v_spac)]
             #bblons = [UL_h - (4*h_spac), UL_h - (4*h_spac), UL_h + (5*h_spac), UL_h + (5*h_spac)]
             tbbox = Polygon(zip(bblons, bblats))
             pixbbox = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[tbbox])
             fig, ax = plt.subplots(figsize=(12, 10))
             pixbbox.plot(ax=ax, color="gray")
             plt.show()
             gedi_bup = gpd.read_file("GEDI_2B_clean/GEDI_2B_clean.shp", bbox = tbbox)
             if gedi_bup.shape[0] > 0:
                 print(gedi_bup.geometry[0].x, gedi_bup.geometry[0].y)
             else:
                 continue
             print(gedi_bup.shape[0])
             #print(gedi_bup.geometry)
             #print(dir(gedi_bup.geometry))
             #for k in range(gedi_bup.shape[0]):
             #    good = True
             #    if gedi_bup.geometry[k].x <= UL_h - (4 * h_spac) or gedi_bup.geometry[k].x = > UL_h + (5 * h_spac):
             ##        # no good
             #        good = False
             #    if gedi_bup.geometry[k].y <= UL_v - (4 * v_spac) or gedi_bup.geometry[k].y = > UL_v + (5 * v_spac):
             #        good = False

             #gedi_bup = gedi_bup.iloc[[2, 3]]
             print("done")
             tlats = [UL_v, UL_v + v_spac, UL_v + v_spac, UL_v]
             tlons = [UL_h, UL_h, UL_h + h_spac, UL_h + h_spac]
             poly = Polygon(zip(tlons, tlats))

             pixpoly = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[poly])
             print("making voronoi diagram")
             #print(list(gedi_bup.columns))
             #print(list(pixpoly.columns))
             #print(type(pixpoly.geometry))
             """"""
             gdf_coords = points_to_coords(gedi_bup.geometry)
             poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(gdf_coords, cascaded_union(pixbbox.geometry))
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

