import os
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
    ix = int((cx - ulh) / psh)
    iy = int((ulv - cy) / psv)
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


"""
# if verbosity > 0:
#    print("progress 0/50 ", end="", flush=True)

prescreen1 = False
extreme_bounds = [-100000, 100000]
prescreen_dist = 35 ** 2
prescreen_forestp = 0.95
if shuffleorder:
    irange_default = np.arange(yrsize[0])
    jrange_default = np.arange(yrsize[1])
    np.random.shuffle(irange_default)
    np.random.shuffle(jrange_default)
else:
    irange_default = np.arange(yrsize[0])
    jrange_default = np.arange(yrsize[1])

### INTERPOLATION PARAMS
xsq_res = 30
ysq_res = 70
xsq_num = 4

file_batch_size = 10

#iterate over potentially shuffled x, y
for i in irange_default:
    for j in jrange_default:

        progress += 1
        temp_lcpercent = 0
        extreme_warning = False
        #if the value at the current y is not the nodata value
        if y_npar[i, j] != yndv:
            #if not h5_mode or (h5_mode and h5_scsv):
            #    if channel_first:
            #        x_img = np.zeros((channels, imgsize + (2 * pad_img), imgsize + (2 * pad_img)))
            #    else:
            #        x_img = np.zeros((imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
            #nlcd_count = 0



            ### Assume csv
            m1_ximg = np.zeros((xsq_num, xsq_num, len(xr_npar)))
            m2_ximg = np.zeros((xsq_num, xsq_num, len(xr_npar)))
            og_ximg = np.zeros((imgsize + (2 * pad_img), imgsize + (2 * pad_img), channels))
            ### we have i, j for y...
            ### build 4x4 grid
            sq_relative = []
            sq_start = [ysq_res/2 - ((xsq_num/2) * xsq_res), ysq_res/2 - ((xsq_num/2) * xsq_res)] # eg 35 - 2*30 = -25/70
            for ii in range(xsq_num):
                sq_relative.append([])
                for jj in range(xsq_num):
                    sq_relative[ii].append([(sq_start[0] + ((ii + 0.5) * xsq_res))/ysq_res,
                                            (sq_start[1] + ((jj + 0.5) * xsq_res))/ysq_res])
            ### so we should have (-10, -10) (20, -10) (etc..)
            ### at each of these points we need the 4 closest in actual crs...?
            ### METHOD 1 - CONVEX COMBO
            ### METHOD 2 - BASIC SAMPLING
            for ii in range(len(sq_relative)):
                for jj in range(len(sq_relative[ii])):
                    ### convert fractional index corresponding to centerpoints of ideal raster grid to crs
                    tempx, tempy = idx_pixctr(sq_relative[ii][jj][0], sq_relative[ii][jj][1], yulh, yulv, ypxh, ypxv, mode='ul')
                    for k in range(len(xr_npar)):
                        ### convert centerpoint coords in crs to index in raster layer (upper left)
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3])
                        ### get centerpoint of raster pixel in crs. Location relative to tempx, tempy will help ...
                        ### ... to determine other 3 closest centerpoints
                        ### stands for rasterxcenter, etc
                        rstxc, rstyc = idx_pixctr(tempi, tempj, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3], mode='ctr')

                        ### get ids for convex combo
                        ### ideal center further left than box it falls in... need to go 1 index left (<)
                        sqxdiff = tempx - rstxc
                        sqydiff = tempy - rstyc
                        if sqxdiff <= 0: #tx - rx < 0 => tx < rx...
                            sq_refi = -1
                        else:
                            sq_refi = 1
                        ### ideal center further up than box it falls in... need to go 1 index up (<)
                        if sqydiff <= 0:
                            sq_refj = -1
                        else:
                            sq_refj = 1
                        sqwidthx = xr_params[k][2]  # x raster pixel width (horizontal)
                        sqwidthy = xr_params[k][3]  # x raster pixel height (vertical)
                        sq_refs = [(tempi, tempj), (tempi+sq_refi, tempj), (tempi, tempj+sq_refj),
                                   (tempi+sq_refi, tempj+sq_refi)]
                        sqweights = [sqxdiff**2 + sqydiff**2, (sqwidthx-sqxdiff)**2 + sqydiff**2,
                                      sqxdiff**2 + (sqwidthy - sqydiff)**2, (sqwidthx-sqxdiff)**2 + (sqwidthy - sqydiff)**2]
                        sqnorm = sum(sqweights)
                        value = 0
                        for sqw in range(4):
                            value += (sqweights[sqw] / sqnorm) * (xr_npar[k][sq_refs[sqw][0], sq_refs[sqw][1]])

                        m1_ximg[ii, jj, k] = value
                        m2_ximg[ii, jj, k] = xr_npar[k][tempi, tempj]

            ### OG METHOD - 5m SAMPLING
            for k in range(len(xr_npar)):
                ### ... Try again with a buffer to get 16x16 image
                for si in range(0 - pad_img, imgsize + pad_img):
                    for sj in range(0 - pad_img, imgsize + pad_img):
                        ### want -.5, .5, 1.5, 2.5, etc...
                        sxoffset = ((2 * si) + 1) / (2 * imgsize)
                        syoffset = ((2 * sj) + 1) / (2 * imgsize)
                        tempx, tempy = idx_pixctr(i + sxoffset, j + syoffset, yulh, yulv, ypxh,
                                                  ypxv, mode='ul')
                        tempi, tempj = coords_idx(tempx, tempy, xr_params[k][0], xr_params[k][1],
                                                  xr_params[k][2], xr_params[k][3])

                        og_ximg[si + pad_img, sj + pad_img, k] = xr_npar[k][tempi, tempj]

            ### make a string of
            ### file name, y value, nsuccess, y raster coordinates, ..., average distance to nearest neighbor
            # good to save
            if not skip_save:
                database.append(["/datasrc/x_img/x_" + str(nsuccess) + ".csv", y_npar[i, j], nsuccess, i, j])
                np.savetxt(fs_loc + "/datasrc/m1_ximg/x_" + str(nsuccess) + ".csv",
                           m1_ximg.reshape(-1, m1_ximg.shape[2]),
                           delimiter=",", newline="\n")
                np.savetxt(fs_loc + "/datasrc/m2_ximg/x_" + str(nsuccess) + ".csv",
                           m2_ximg.reshape(-1, m2_ximg.shape[2]),
                           delimiter=",", newline="\n")
                np.savetxt(fs_loc + "/datasrc/og_ximg/x_" + str(nsuccess) + ".csv",
                           og_ximg.reshape(-1, og_ximg.shape[2]),
                           delimiter=",", newline="\n")
            nsuccess += 1
            if verbosity > 0 and nsuccess % (testmode // 50) == 0:
                print("-", end="", flush=True)
                # else:
                #    dbins = list(failsafe_copy)
            if testmode > 0 and nsuccess > testmode:
                print()
                print("max ring size: ", maxringsize)
                print("avg ring size: ", avgringsize // nsuccess)
                print("saving ydata")
                ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
                ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
                # save h5set
                # print("max ring size: ", maxringsize)
                print("extreme encounter report:")
                no_enc = True
                for i in range(len(extreme_encounter)):
                    if extreme_encounter[i] > 0:
                        no_enc = False
                        print(" ", i, extreme_encounter[i])
                if no_enc:
                    print("no extreme encounters")

                plt.figure()
                plt.bar(diids, dists)
                plt.title("distribution of nlcd values over 5m regions / sample")
                plt.savefig("../figures/nlcd_dist_.png")
                plt.cla()
                plt.close()

                for i in range(len(dists)):
                    if dists[i] != 0:
                        dists[i] = math.log(dists[i])
                plt.figure()
                plt.bar(diids, dists)
                plt.title("log distribution of nlcd values over 5m regions / sample")
                plt.savefig("../figures/nlcd_dist_log.png")
                plt.cla()
                plt.close()

                ### dist things
                # rasters_names_list = ["srtm", "nlcd", "slope", "aspect", "ecostress_WUE"]
                # for idb in range(len(dbins)):
                #    plt.figure()
                #    plt.bar(dbins[idb])
                #    plt.title(rasters_names_list[idb] + " distribution over 5m pixels")
                #    plt.savefig("../figures/pixel_distributions/" + rasters_names_list[idb] + "_dbn.png")
                #    plt.cla()
                #    plt.close()

                sys.exit("exiting after testmode samples")

print()
# print(maxringsize)

#print("max ring size: ", maxringsize)
#print("saving ydata")
#ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
#ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")

print("max ring size: ", maxringsize)
print("avg ring size: ", avgringsize // nsuccess)
print("saving ydata")
ydataframe = pd.DataFrame(data=database, columns=pd_colnames)
ydataframe.to_csv(fs_loc + "/datasrc/ydata.csv")
# save h5set
if h5_mode:
    ###need to make sure last chunk is saved
    print("saving last h5 chunk...")
    h5dset.resize(h5len, axis=0)
    h5dset[h5len - h5tid:h5len, :, :, :] = h5_chunk[:h5tid, :, :, :]
    print("saving h5 dset...")
    h5_dataset.close()
# print("max ring size: ", maxringsize)
print("extreme encounter report:")
no_enc = True
for i in range(len(extreme_encounter)):
    if extreme_encounter[i] > 0:
        no_enc = False
        print(" ", i, extreme_encounter[i])
if no_enc:
    print("no extreme encounters")

plt.figure()
plt.bar(diids, dists)
plt.title("distribution of nlcd values over 5m regions / sample")
plt.savefig("../figures/nlcd_dist_.png")
plt.cla()
plt.close()

for i in range(len(dists)):
    if dists[i] != 0:
        dists[i] = math.log(dists[i])
plt.figure()
plt.bar(diids, dists)
plt.title("log distribution of nlcd values over 5m regions / sample")
plt.savefig("../figures/nlcd_dist_log.png")
plt.cla()
plt.close()
"""