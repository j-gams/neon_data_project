### Written by Jerry Gammie @j-gams

import numpy as np
import geopandas as gpd
#from geovoronoi import voronoi_regions_from_coords, points_to_coords


print("loading aoi")
#aoi = gpd.read_file("../raw_data/neon_aoi/NEON_3D_Boundary.shp")
print("loading gedi centroids")
points = gpd.read_file("../raw_data/gedi_pts/GEDI_2B_clean.shp")

print(points.head)
"""
aoi_shape = aoi.geometry
print(type(aoi_shape))

coords = points_to_coords(points)
print("generating diagram")
region_polys, region_pts = voronoi_regions_from_coords(coords, area_shape)
"""