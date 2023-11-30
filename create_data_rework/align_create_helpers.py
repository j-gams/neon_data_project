import os
import math
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

def load_config():
    pass

def make_pt_indexer(critical_fields, x_point_file, n_x_point_files, file_location, index_name=""):
    point_layers = []
    point_layer_names = []
    point_layer_indexer = {}
    ### TODO -- this system seems to assume only 1 pointfile
    for i in range(len(critical_fields)):
        point_layer_idx = 0
        for j in range(len(x_point_file.fields)):
            if critical_fields[i] in x_point_file.fields[j][0]:
                point_layer_indexer[len(point_layers)] = (n_x_point_files-1, i, j, point_layer_idx)
                point_layers.append(critical_fields[i] + "_" + str(point_layer_idx))
                point_layer_names.append(x_point_file.fields[j][0])
                point_layer_idx += 1

    print("writing point indexer file")
    if os.path.exists(file_location + "/point_reformat/pt_indexer_util" + index_name + ".txt"):
        os.system("rm " + file_location + "/point_reformat/pt_indexer_util" + index_name + ".txt")
    for layer_key in point_layer_indexer.keys():
        layer_key_str = ""
        for elt in point_layer_indexer[layer_key]:
            layer_key_str += str(elt) + ","
        os.system('echo "' + str(layer_key) + ':' + layer_key_str[:-1] + '" >> '
                  + file_location + '/point_reformat/pt_indexer_util.txt')
    return point_layers, point_layer_names, point_layer_indexer

def reformat_helper(params):
    #iter, point_record = params
    if params[0] == 0:
        return coordinate_reformat_helper(params[1])
    else:
        return field_reformat_helper(params)

def coordinate_reformat_helper(point_record):
    record_string = "lon,lat\n"
    for i in range(len(point_record)):
        a, b = point_record[i].record[3], point_record[i].record[2]
        record_string += str(a) + "," + str(b) + "\n"
    return record_string

def field_reformat_helper(params):
    real_iter, x_point_file, point_record, critical_field = params
    fields_ids = []
    fields_names = ""
    record_string = ""
    for k in range(len(x_point_file.fields)):
        if critical_field in x_point_file.fields[k][0]:
            fields_ids.append(k)
            fields_names += x_point_file.fields[k][0] + ","
    record_string += fields_names[:-1] + "\n"
    for j in range(len(point_record)):
        dumpstr = ""
        for k in fields_ids:
            if critical_field in x_point_file.fields[k][0]:
                dumpstr += str(point_record[j].record[k]) + ","
        record_string += dumpstr[:-1] + "\n"
    return record_string

def sample_helper():
    pass


# in geotif, raster data stored from upper left pixel coordinates, pixel width, and rotation.
# so we can get value at coordinate (x,y) from looking at pixel at
# (x - UL_x)/size_x
# (UL_y - y)/size_y
# here these are switched because in the geographic crs, increasing coordinates go up and to the right
# however within the matrix they go down and to the right, so the sign must be reversed for y

# turn coordinates into an index in the data array
def coords_idx(cx, cy, ulh, ulv, psh, psv):
    ix = (cx - ulh) / psh
    iy = (ulv - cy) / psv
    return ix, iy

def crs_switch(cx, cy, fromcrs, tocrs):
    from_ref = osr.SpatialReference(wkt=fromcrs)
    to_ref = osr.SpatialReference(wkt=tocrs)
    transform = osr.CoordinateTransformation(from_ref, to_ref)

    coords = transform.TransformPoint(cx, cy)
    return coords[0:2]

# get coordinates of pixel from index
# if mode is 'ctr': get coords of center of pixel
# else if mode is 'ul': get coords of upper left of pixel
def idx_pixctr(ix, iy, ulh, ulv, psh, psv, mode='ul'):
    offsetx = 0
    offsety = 0
    if mode == 'ctr':
        offsetx = psh / 2
        offsety = psv / 2
    cx = ulh + (ix * psh) + offsetx
    cy = ulv - (iy * psv) + offsety
    return cx, cy

def cdist(x1, y1, x2, y2):
    return (((x1 - x2) ** 2) + ((y1 - y2) ** 2))

### stupid k nearest
def getkclose(shapes, centerx, centery, k, ulh, ulv, psh, psv):
    distlist = []
    ids = []
    cx, cy = idx_pixctr(0.5 + centerx, 0.5 + centery, ulh, ulv, psh, psv, mode='ul')
    for i in range(len(shapes)):
        a = shapes[i, 0]
        b = shapes[i, 1]
        distlist.append(cdist(a, b, cx, cy))
        ids.append(i)
    # sort ids by distlist
    ids = [id for _, id in sorted(zip(distlist, ids), key=lambda pair: pair[0])]
    return ids[:k]

def mpalign(glob_params, xr_npar1, band):
    grid_size, temp_dsampler, temp_brfrac, ycrs, xcrs, temp_frac, nodata_ignore, nodata_oops, avg_method = glob_params

    oobcounter = 0
    all_oob = 0
    temp_data = np.zeros(grid_size[1])
    ### start 1-band alignment
    ### first iteration is just band


    weighterj = np.ones((1, temp_dsampler))
    weighteri = np.ones((temp_dsampler, 1))

    for j in range(grid_size[1]):
        #ul of window in window index
        yi = band #* temp_brfrac
        yj = j #* temp_brfrac

        yulh, yulv, ypxh, ypxv = ycrs
        xulh, xulv, xpxh, xpxv = xcrs
        #idx_offset = 0

        coordyi, coordyj = idx_pixctr(yi, yj, yulh, yulv, ypxh, ypxv)

        xcrs_yi, xcrs_yj = coords_idx(coordyi, coordyj, xulh, xulv, xpxh, xpxv)
        #print("**", yi, yj)
        #print(xcrs_yi, xcrs_yj)
        weighteri[0, 0] = xcrs_yi % 1
        weighterj[0, 0] = xcrs_yj % 1
        weighteri[-1, 0] = (temp_frac - weighteri[0, 0]) % 1
        weighterj[0,-1] = (temp_frac - weighterj[0, 0]) % 1

        oobmask = inbounds2(xcrs_yi, xcrs_yj, temp_frac, xr_npar1, nodata_ignore, temp_dsampler)
        countxy = np.where(oobmask != 0)
        temp = xr_npar1[countxy[0]+math.floor(xcrs_yi), countxy[1]+math.floor(xcrs_yj)]
        if len(temp) == 0:
            all_oob += 1
            temp_data[j] = nodata_oops
        if avg_method == "mean":
            temp_data[j] = np.mean(temp * (weighteri @ weighterj)[countxy])
        elif avg_method == "mode":
            res2 = np.unique(temp)
            res3 = np.bincount(np.searchsorted(res2, temp), (weighteri @ weighterj)[countxy])
            temp_data[j] = res2[res3.argmax()]

        return temp_data, all_oob
def badinbounds(val, grid):
    ix, iy = val
    if ix >= 0 and iy >= 0 and ix < grid.shape[0] and iy < grid.shape[1]:
        return True
    return False
def badmpalign(glob_params, xr_npar1, band):
    grid_size, temp_dsampler, temp_brfrac, ycrs, xcrs, temp_frac, nodata_ignore, nodata_oops, avg_method = glob_params

    temp_slice = max(70 // 10, 1)
    temp_factor = max(30 // 10, 1)
    minioffset = (1/(2 * temp_factor))

    oobcounter = 0
    all_oob = 0
    temp_data = np.zeros(grid_size[1])
    for j in range(grid_size[1]):
        #ul of window in window index
        yi = band #* temp_brfrac
        yj = j #* temp_brfrac

        yulh, yulv, ypxh, ypxv = ycrs
        xulh, xulv, xpxh, xpxv = xcrs
        idx_offset = 0

        coordyi, coordyj = idx_pixctr(yi, yj, yulh, yulv, ypxh, ypxv)

        xcrs_yi, xcrs_yj = coords_idx(coordyi, coordyj, xulh, xulv, xpxh, xpxv)

        temp = []
        for ii in range(temp_slice):
            for jj in range(temp_slice):
                deci = ii / temp_factor + minioffset
                decj = jj / temp_factor + minioffset
                if badinbounds((xcrs_yi + deci, xcrs_yj + decj), xr_npar1):
                    #print(xcrs_yi + deci, xcrs_yj + decj)
                    temptemp = xr_npar1[int(xcrs_yi + deci), int(xcrs_yj + decj)]
                    if temptemp != nodata_ignore:
                        temp.append(temptemp)
                else:
                    oobcounter += 1
        if len(temp) == 0:
            all_oob += 1
            temp_data[j] = nodata_oops
        else:
            temp = np.array(temp)
            if avg_method == "mean":
                temp_data[j] = np.mean(temp)
            elif avg_method == "mode":
                vals, counts = np.unique(temp, return_counts=True)
                temp_data[j] = vals[np.argwhere(counts == np.max(counts))][0]
    return temp_data, all_oob, oobcounter

def inbounds2(uli, ulj, dim, grid, ndval, temp_dsampler):
    oobmask = np.ones((temp_dsampler, temp_dsampler))
    mini = 0
    maxi = temp_dsampler
    minj = 0
    maxj = temp_dsampler
    if uli < 0:
        mini = -math.floor(uli)
        oobmask[mini,:] = 0
    if ulj < 0:
        minj = -math.floor(ulj)
        oobmask[:,minj] = 0
    if uli + dim >= grid.shape[0]:
        print("dim", uli, dim, grid.shape[0])
        maxi = temp_dsampler - math.ceil(uli + dim - grid.shape[0])
        oobmask[maxi, :] = 0
    if ulj + dim >= grid.shape[1]:
        maxj = temp_dsampler - math.ceil(ulj + dim - grid.shape[1])
        oobmask[:, maxj] = 0
    #print("basevals", mini+int(uli), maxi+int(uli), minj+int(ulj), maxj+int(ulj))
    nn = np.where(grid[mini+int(uli): maxi+int(uli), minj+int(ulj): maxj+int(ulj)] == ndval)
    oobmask[(nn[0] + mini, nn[1] + minj)] = 0

    return oobmask

class alignment_sampling:
    def __init__ (self, grid_res, base_grid_size, base_grid_res, basecrs, selfcrs, params):
        self.grid_res = grid_res
        self.base_res = base_grid_res
        self.base_size = base_grid_size
        total_m = (self.base_size[0]*self.base_res, self.base_size[1]*self.base_res)
        self.grid_size = ((total_m[0] // grid_res) + 1, (total_m[1] // grid_res) + 1)

        print(self.grid_size)
        self.ycrs, self.yproj = basecrs
        self.xcrs, self.xproj = selfcrs
        print(self.ycrs)
        print(self.xcrs)

        print(self.yproj)
        print(self.xproj)

        self.data = np.zeros(self.grid_size)

        self.px_mode = params[0]


    def align(self, xr_npar, subset=-1):
        oobcounter = 0
        boundi = self.grid_size[0] - 1
        boundj = self.grid_size[1] - 1
        if subset != -1:
            boundi = subset
            boundj = subset
        for i in range(boundi):
            for j in range(boundj):
                # m/70
                yi = (i*self.grid_res)/self.base_res
                yj = (j*self.grid_res)/self.base_res

                yulh, yulv, ypxh, ypxv = self.ycrs
                xulh, xulv, xpxh, xpxv = self.xcrs

                #incorporate ul, ctr, etc
                idx_offset = 0
                if self.px_mode == "ctr":
                    idx_offset = self.grid_res/(2*self.base_res)

                # convert i, j to coord in y-grid
                ycrs_x, ycrs_y = idx_pixctr(yi+idx_offset, yj+idx_offset,
                                            yulh, yulv, ypxh, ypxv)
                #print(ycrs_x, ycrs_y)
                # convert coord-to-coord
                #xcrs_x, xcrs_y = crs_switch(ycrs_x, ycrs_y, self.yproj, self.xproj)

                #print(xcrs_x, xcrs_y)
                # convert coord to x-idx
                xi, xj = coords_idx(ycrs_x, ycrs_y, xulh, xulv, xpxh, xpxv)

                # sample at location
                try:
                    self.data[i, j] = xr_npar[xi, xj]
                except:
                    oobcounter += 1
        print(oobcounter)
        print(np.min(self.data), np.max(self.data))
        self.newcrs = (yulh, ypxh * (self.grid_res/self.base_res), 0,
                       yulv, 0, ypxv * (-self.grid_res/self.base_res))
        self.newproj = self.yproj

class alignment_average:
    def __init__ (self, grid_res, base_grid_size, base_grid_res, basecrs, selfcrs, params, nodata):
        ### crs res in meters, eg 30
        self.grid_res = grid_res
        ### base crs res in meters, eg 70
        self.base_res = base_grid_res
        ### size of base crs - x, y length of np array
        self.base_size = base_grid_size
        ### total meters - grid l * res, grid w * res
        total_m = (self.base_size[0]*self.base_res, self.base_size[1]*self.base_res)
        ### grid size - total meters // layer res meters
        self.grid_size = ((total_m[0] // grid_res) + 1, (total_m[1] // grid_res) + 1)
        ### nd value
        self.nodata_ignore = nodata
        self.nodata_oops = -99999

        print(self.grid_size)
        self.ycrs, self.yproj = basecrs
        self.xcrs, self.xproj = selfcrs

        self.data = np.zeros(self.grid_size) + self.nodata_oops

        ### in ("mean", "mediam", "mode"
        self.avg_method = params[0]
        ### sample at this res within crs grid sq
        self.sample_to = params[1]
        ### expect res from current raster
        self.expect_res = params[2]
        if len(params) > 3:
            self.px_mode = params[3]
        else:
            self.px_mode = "ul"

        self.temp_slice = max(self.grid_res // self.sample_to, 1)
        self.temp_factor = max(self.expect_res // self.sample_to, 1)
        self.temp_dsampler = (self.grid_res // self.expect_res) + 2
        self.temp_frac = self.grid_res/self.expect_res
        self.temp_brfrac = self.grid_res / self.base_res

        self.coordbasei = np.zeros((self.temp_dsampler, self.temp_dsampler))
        self.coordbasej = np.zeros((self.temp_dsampler, self.temp_dsampler))
        for i in range(self.temp_dsampler):
            for j in range(self.temp_dsampler):
                self.coordbasei[i, j] = i
                self.coordbasej[i, j] = j
        #self.coordbasei = self.coordbasei.flatten()
        #self.coordbasej = self.coordbasej.flatten()

    def inbounds(self, val, grid):
        ix, iy = val
        if ix >= 0 and iy >= 0 and ix < grid.shape[0] and iy < grid.shape[1]:
            return True
        return False


    def mpimport(self, datimport):
        yulh, yulv, ypxh, ypxv = self.ycrs
        self.newcrs = (yulh, ypxh * (self.grid_res / self.base_res), 0,
                       yulv, 0, ypxv * (-self.grid_res / self.base_res))
        self.newproj = self.yproj
        self.newndv = self.nodata_oops
        self.data = np.array(datimport)
        print(self.data.shape)
        print("datamin", np.min(self.data))
        print("datamax", np.max(self.data))
        print("complete")

    def alignbasic(self, xr_npar, subset=-1):
        all_oob = 0

        ### grid_res is target grid res in m
        ### base_res is base res in m
        ### base_size is shape of base np array
        ### total_m is dimensions of raster in m
        ### grid size is total_meters over target grid size
        boundi = self.grid_size[0]
        boundj = self.grid_size[1]
        if subset != -1:
            boundi = subset
            boundj = subset
        oobcounter = 0
        nodcounter = 0
        print("alignment starting...")
        temp_slice = max(self.grid_res // self.sample_to, 1)
        temp_factor = max(self.expect_res // self.sample_to, 1)
        for i in range(boundi):
            if (i+1) % (boundi//100) == 0:
                print("1%")
            for j in range(boundj):
                yi = (i * self.grid_res) / self.base_res
                yj = (j * self.grid_res) / self.base_res
                yulh, yulv, ypxh, ypxv = self.ycrs
                xulh, xulv, xpxh, xpxv = self.xcrs
                idx_offset = 0
                ### convert to coords
                ycrs_x, ycrs_y = idx_pixctr(yi + idx_offset, yj + idx_offset,
                                            yulh, yulv, ypxh, ypxv)
                ### convert to fractional index in og grid
                xi, xj = coords_idx(ycrs_x, ycrs_y, xulh, xulv, xpxh, xpxv)
                temp = [] #np.zeros((temp_slice, temp_slice))

                #oobflag = 0
                for ii in range(temp_slice):
                    for jj in range(temp_slice):
                        deci = ii//temp_factor
                        decj = jj//temp_factor
                        if self.inbounds((xi + deci, xj + decj), xr_npar):
                            temptemp = xr_npar[int(xi + deci), int(xj + decj)]
                            if temptemp != self.nodata_ignore:
                                temp.append(temptemp)
                        else:
                            oobcounter += 1
                            #oobflag += 1
                #if oobflag == temp_slice * temp_slice:
                #    print("all oob")
                if len(temp) == 0:
                    temp = [self.nodata_oops]
                    all_oob += 1
                temp = np.array(temp)
                if self.avg_method == "mean":
                    self.data[i, j] = np.mean(temp)
                elif self.avg_method == "mode":
                    vals, counts = np.unique(temp, return_counts=True)
                    self.data[i, j] = vals[np.argwhere(counts == np.max(counts))][0]
        print("oobs", oobcounter)
        print("all oobs", all_oob)
        print("all samples",  boundi*boundj)
        print(np.min(self.data), np.max(self.data))
        self.newcrs = (yulh, ypxh * (self.grid_res / self.base_res), 0,
                       yulv, 0, ypxv * (-self.grid_res / self.base_res))
        self.newproj = self.yproj
        self.newndv = self.nodata_oops

    def align2(self, xr_npar, subset=-1):
        oobcounter = 0
        ### sample... ultimate resolution // sampling resolution (7x7), eg
        temp_slice = self.grid_res // self.sample_to
        ### expected res of original raster // sampling resolution -- samples per og square

        temp_factor = self.expect_res // self.sample_to
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # meter location / window grid - finding location in window-aligned tile
                yi = (i * self.grid_res) / self.base_res
                yj = (j * self.grid_res) / self.base_res
                yulh, yulv, ypxh, ypxv = self.ycrs
                xulh, xulv, xpxh, xpxv = self.xcrs
                idx_offset = 0
                # convert i, j from idx in ultimate grid to coord
                ycrs_x, ycrs_y = idx_pixctr(yi + idx_offset, yj + idx_offset,
                                            yulh, yulv, ypxh, ypxv)
                # convert coord to x-idx
                xi, xj = coords_idx(ycrs_x, ycrs_y, xulh, xulv, xpxh, xpxv)
                #temp = np.zeros(temp_size)



    def align(self, xr_npar, subset=-1):
        oobcounter = 0
        boundi = self.grid_size[0] - 1
        boundj = self.grid_size[1] - 1
        if subset != -1:
            boundi = subset
            boundj = subset
        temp_size = (self.grid_res // self.sample_to, self.grid_res // self.sample_to)
        temp_slice = self.grid_res // self.sample_to
        temp_factor = self.expect_res // self.sample_to
        for i in range(boundi):
            for j in range(boundj):
                # m/70
                yi = (i*self.grid_res)/self.base_res
                yj = (j*self.grid_res)/self.base_res

                yulh, yulv, ypxh, ypxv = self.ycrs
                xulh, xulv, xpxh, xpxv = self.xcrs

                #incorporate ul, ctr, etc
                idx_offset = 0
                if self.px_mode == "ctr":
                    idx_offset = self.grid_res/(2*self.base_res)

                # convert i, j to coord in y-grid
                ycrs_x, ycrs_y = idx_pixctr(yi+idx_offset, yj+idx_offset,
                                            yulh, yulv, ypxh, ypxv)

                # convert coord to x-idx
                # UL within
                xi, xj = coords_idx(ycrs_x, ycrs_y, xulh, xulv, xpxh, xpxv)
                # sample at location
                temp = np.zeros(temp_size)
                try:
                    ### need more thought here
                    for ii in range(temp_slice):
                        for jj in range(temp_slice):
                            temp[ii, jj] = xr_npar[int(xi+(ii//temp_factor)),
                                                   int(xj+(jj//temp_factor))]

                    if self.avg_method == "mean":
                        self.data[i, j] = np.mean(temp)
                    elif self.avg_method == "mode":
                        vals, counts = np.unique(temp, return_counts=True)
                        self.data[i, j] = np.argwhere(counts == np.max(counts))
                    elif self.avg_method == "median":
                        self.data[i, j] = np.median(temp)
                except:
                    oobcounter += 1
        print(oobcounter)
        print(np.min(self.data), np.max(self.data))
        self.newcrs = (yulh, ypxh * (self.grid_res/self.base_res), 0,
                       yulv, 0, ypxv * (-self.grid_res/self.base_res))
        self.newproj = self.yproj