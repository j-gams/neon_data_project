import sys
import rioxarray as rxr
import xarray as xr

# cmd fname nodata_value
if len(sys.argv) < 3:
    print("insufficient arguments provided")
else:
    fname = sys.argv[1]
    nd_val = float(sys.argv[2])
    if nd_val == int(nd_val):
        nd_val = int(nd_val)
    rxr_data = rxr.open_rasterio(fname, masked=True).squeeze()
    print("previous no_data value for", fname.split("/")[-1], ":", rxr_data.rio.nodata)
    print("setting no_data value to", nd_val)
    rxr_data.rio.set_nodata(nd_val, inplace=True)
    rxr_data.rio.to_raster(fname)