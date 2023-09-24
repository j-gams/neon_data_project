Last login: Mon Aug 28 11:36:55 on ttys006
(base) jerry@lumon ~ % ssh 128.138.224.159
jerry@128.138.224.159's password:
Welcome to Ubuntu 22.04.1 LTS (GNU/Linux 5.19.0-45-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

 * Introducing Expanded Security Maintenance for Applications.
   Receive updates to over 25,000 software packages with your
   Ubuntu Pro subscription. Free for personal use.

     https://ubuntu.com/pro

242 updates can be applied immediately.
1 of these updates is a standard security update.
To see these additional updates run: apt list --upgradable

*** System restart required ***
Last login: Mon Aug 28 23:48:27 2023 from 198.11.30.103
(base) jerry@woolley:~$ cd work/earthlab/munge_data/create_data/
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls
alt_datacube.py          data_nn_gedi                 __pycache__
analyze_clipped.py       dat_obj.py                   readme.md
build_train_val_test.py  dcube_dist_analysis.py       reset_raster_nd.py
check_clip.py            h5_sanitycheck.py            rfdata_loader.py
code_match.py            interpolation_comparison.py  testbed.py
corr_analysis.py         match_create_set.py          test_vnoi.py
create_2.py              mcs_interpolation.py         test_xdata.py
create_data.sh           nohup.out                    tif_merge_convert.py
datacube_set.py          pixelstats_lm.py             voronator.py
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls -al
total 420
drwxrwxr-x  4 jerry jerry  4096 Aug 29 00:13 .
drwxrwxr-x 11 jerry jerry  4096 Jun  2 05:30 ..
-rw-rw-r--  1 jerry jerry 23796 Jun  2 08:34 alt_datacube.py
-rw-rw-r--  1 jerry jerry 14660 Sep 15  2022 analyze_clipped.py
-rw-rw-r--  1 jerry jerry  2350 Sep 14  2022 build_train_val_test.py
-rw-rw-r--  1 jerry jerry  1443 Sep 13  2022 check_clip.py
-rw-rw-r--  1 jerry jerry  6390 Sep 13  2022 code_match.py
-rw-rw-r--  1 jerry jerry  2951 Sep 13  2022 corr_analysis.py
-rw-rw-r--  1 jerry jerry 52631 Dec 13  2022 create_2.py
-rw-rw-r--  1 jerry jerry  1762 Jul 13  2022 create_data.sh
-rw-rw-r--  1 jerry jerry 23358 Oct 26  2022 datacube_set.py
drwxrwxr-x  4 jerry jerry  4096 Jun 21  2022 data_nn_gedi
-rw-rw-r--  1 jerry jerry  6838 Jun  2 08:45 dat_obj.py
-rw-rw-r--  1 jerry jerry  5140 Nov  4  2022 dcube_dist_analysis.py
-rw-rw-r--  1 jerry jerry  2195 Sep 13  2022 h5_sanitycheck.py
-rw-rw-r--  1 jerry jerry  2675 Jun  2 08:16 interpolation_comparison.py
-rw-rw-r--  1 jerry jerry 44202 Sep 19  2022 match_create_set.py
-rw-rw-r--  1 jerry jerry 40098 Jun  2 08:08 mcs_interpolation.py
-rw-------  1 jerry jerry 58289 Sep 11  2022 nohup.out
-rw-rw-r--  1 jerry jerry  1359 Sep 19  2022 pixelstats_lm.py
drwxrwxr-x  2 jerry jerry  4096 Jun  2 08:45 __pycache__
-rw-rw-r--  1 jerry jerry 55778 Oct 26  2022 readme.md
-rw-rw-r--  1 jerry jerry  1256 Sep 13  2022 reset_raster_nd.py
-rw-rw-r--  1 jerry jerry  1302 Sep 13  2022 rfdata_loader.py
-rw-rw-r--  1 jerry jerry   196 Sep 13  2022 testbed.py
-rw-rw-r--  1 jerry jerry  4056 Sep 13  2022 test_vnoi.py
-rw-rw-r--  1 jerry jerry  2430 Sep 13  2022 test_xdata.py
-rw-rw-r--  1 jerry jerry  4170 Sep 13  2022 tif_merge_convert.py
-rw-rw-r--  1 jerry jerry   549 Sep 13  2022 voronator.py
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim interpolation_comparison.py

(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim mcs_interpolation.py
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ cd ..
(base) jerry@woolley:~/work/earthlab/munge_data$ ls
 create_data   data   experiments   figures   logs   models   raw_data   README.md   ssh_transfer.txt  ' .txt'
(base) jerry@woolley:~/work/earthlab/munge_data$ cd data/data_interpolated
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated$ ls
datasrc  meta  point_reformat
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated$ ls -al
total 20
drwxrwxr-x  5 jerry jerry 4096 Aug 29 00:14 .
drwxrwxr-x 10 jerry jerry 4096 Aug 29 00:15 ..
drwxrwxr-x  3 jerry jerry 4096 Aug 29 00:14 datasrc
drwxrwxr-x  2 jerry jerry 4096 Aug 29 00:14 meta
drwxrwxr-x  2 jerry jerry 4096 Aug 29 00:14 point_reformat
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated$ cd datasrc/
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated/datasrc$ ls
og_ximg
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated/datasrc$ cd og_ximg/
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated/datasrc/og_ximg$ ls
(base) jerry@woolley:~/work/earthlab/munge_data/data/data_interpolated/datasrc/og_ximg$ cd ../../..
(base) jerry@woolley:~/work/earthlab/munge_data/data$ ls
data_h51     data_interpolated   data_minimode1  minidata       readme.md
data_h5test  data_interpolated1  interpol        minidata_nosa
(base) jerry@woolley:~/work/earthlab/munge_data/data$ cd ../create_data/
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls
alt_datacube.py          create_data.sh               match_create_set.py   rfdata_loader.py
analyze_clipped.py       datacube_set.py              mcs_interpolation.py  testbed.py
build_train_val_test.py  data_nn_gedi                 nohup.out             test_vnoi.py
check_clip.py            dat_obj.py                   pixelstats_lm.py      test_xdata.py
code_match.py            dcube_dist_analysis.py       __pycache__           tif_merge_convert.py
corr_analysis.py         h5_sanitycheck.py            readme.md             voronator.py
create_2.py              interpolation_comparison.py  reset_raster_nd.py
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim mcs_interpolation.py
(base) jerry@woolley:~/work/earthlab/munge_data/create_data$ cd ../models/
(base) jerry@woolley:~/work/earthlab/munge_data/models$ ls
custom_models   mutils.py   nohup.out    saved_models   train_1.py   train_frame.py
logger.py       no2hup.out  __pycache__  test2model.py  train_3.py   train_noise.py
model_train.py  no3hup.out  readme.md    test_model.py  trainer2.py
(base) jerry@woolley:~/work/earthlab/munge_data/models$ ls -al
total 296
drwxrwxr-x  5 jerry jerry   4096 Jun  2 08:48 .
drwxrwxr-x 11 jerry jerry   4096 Jun  2 05:30 ..
drwxrwxr-x  3 jerry jerry   4096 Jun  2 04:51 custom_models
-rw-rw-r--  1 jerry jerry   6869 Sep 13  2022 logger.py
-rw-rw-r--  1 jerry jerry   2644 Sep 13  2022 model_train.py
-rw-rw-r--  1 jerry jerry   4235 Sep 19  2022 mutils.py
-rw-rw-r--  1 jerry jerry  11222 Sep 18  2022 no2hup.out
-rw-rw-r--  1 jerry jerry  10777 Sep 18  2022 no3hup.out
-rw-------  1 jerry jerry 142486 Mar 11 01:58 nohup.out
drwxrwxr-x  2 jerry jerry   4096 Oct 31  2022 __pycache__
-rw-rw-r--  1 jerry jerry   3269 Sep 12  2022 readme.md
drwxrwxr-x 79 jerry jerry  12288 Jun  2 08:49 saved_models
-rw-rw-r--  1 jerry jerry      0 Jul 13  2022 test2model.py
-rw-rw-r--  1 jerry jerry   3508 Sep 13  2022 test_model.py
-rw-rw-r--  1 jerry jerry   8549 Sep 18  2022 train_1.py
-rw-rw-r--  1 jerry jerry   8037 Oct 26  2022 train_3.py
-rw-rw-r--  1 jerry jerry  18566 Jun  2 08:48 trainer2.py
-rw-rw-r--  1 jerry jerry  18684 Mar 10 15:13 train_frame.py
-rw-rw-r--  1 jerry jerry   9886 Sep 21  2022 train_noise.py
(base) jerry@woolley:~/work/earthlab/munge_data/models$ vim trainer2
(base) jerry@woolley:~/work/earthlab/munge_data/models$ vim trainer2.py
(base) jerry@woolley:~/work/earthlab/munge_data/models$ conda activate neon
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['slope_clipped'], ['aspct_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 4)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 4)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 4)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['slope_clipped'], ['aspct_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 4)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 4)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 4)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['slope_clipped'], ['aspct_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (16, 16, 68)
initializing datafold: train set 0
datacube dimensions:  (16, 16, 68)
initializing datafold: validation set 0
datacube dimensions:  (16, 16, 68)
sending rfr_t1 to be trained
generating training log
* BEGINNING FOLD 0
(6930, 4)
(2970, 4)
0.024552814453594595 0.11594380915464006
adding [0.024552814453594595, 0.11594380915464006, 5.908228397369385, 1693290928.9935496]
/home/jerry/work/earthlab/munge_data/models/mutils.py:93: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax1.set_xticklabels(xax1, rotation=45)
/home/jerry/work/earthlab/munge_data/models/mutils.py:94: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax1.set_yticklabels(yax1, rotation=45)
* FOLD 0 COMPLETED. SAVING MODEL...
* DONE
* MODEL SUMMARY
header:  rfr_t1_2023-08-29_00:36:18.916497  rfr_t1
data:       fold   mean_squared_error     mean_absolute_error   train_realtime      train_processtime
fold_0:     0      0.024552814453594595   0.11594380915464006   5.908228397369385   1693290928.9935496
averages:   -      0.024552814453594595   0.11594380915464006   5.908228397369385   1693290928.9935496
minimums:   -      0.024552814453594595   0.11594380915464006   5.908228397369385   1693290928.9935496
min fold:   -      0                      0                     0                   0
maximums:   -      0.024552814453594595   0.11594380915464006   5.908228397369385   1693290928.9935496
max fold:   -      0                      0                     0                   0
* TRAINED WITH HYPERPARAMETERS
hparams:
  model_name:     rfr_test
  save_location:  rfr_t1_2023-08-29_00:36:18.916497
  dropout:        {'mode': 'keep', 'channels': [0, 1, 2, 3]}
  n_estimators:   1000
  max_depth:      None
  n_jobs:         -1
* TRAINED WITH PARAMETERS
tparams:
  folds:        1
  metrics:      ['mean_squared_error', 'mean_absolute_error', 'train_realtime', 'train_processtime']
  mode:         train
  load_from:    na
  save_models:  True
done writing readable log
done training model
generating training log
* BEGINNING FOLD 0
(6930, 4)
(2970, 4)
0.025416168505779647 0.11758764486906559
adding [0.025416168505779647, 0.11758764486906559, 5.632736444473267, 1693290902.0574605]
/home/jerry/work/earthlab/munge_data/models/mutils.py:93: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax1.set_xticklabels(xax1, rotation=45)
/home/jerry/work/earthlab/munge_data/models/mutils.py:94: UserWarning: FixedFormatter should only be used together with FixedLocator
  ax1.set_yticklabels(yax1, rotation=45)
* FOLD 0 COMPLETED. SAVING MODEL...
* DONE
* MODEL SUMMARY
header:  rfr_t1_2023-08-29_00:36:30.946841  rfr_t1
data:       fold   mean_squared_error     mean_absolute_error   train_realtime      train_processtime
fold_0:     0      0.025416168505779647   0.11758764486906559   5.632736444473267   1693290902.0574605
averages:   -      0.025416168505779647   0.11758764486906559   5.632736444473267   1693290902.0574605
minimums:   -      0.025416168505779647   0.11758764486906559   5.632736444473267   1693290902.0574605
min fold:   -      0                      0                     0                   0
maximums:   -      0.025416168505779647   0.11758764486906559   5.632736444473267   1693290902.0574605
max fold:   -      0                      0                     0                   0
* TRAINED WITH HYPERPARAMETERS
hparams:
  model_name:     rfr_test
  save_location:  rfr_t1_2023-08-29_00:36:30.946841
  dropout:        {'mode': 'keep', 'channels': [0, 1, 2, 3]}
  n_estimators:   1000
  max_depth:      None
  n_jobs:         -1
* TRAINED WITH PARAMETERS
tparams:
  folds:        1
  metrics:      ['mean_squared_error', 'mean_absolute_error', 'train_realtime', 'train_processtime']
  mode:         train
  load_from:    na
  save_models:  True
done writing readable log
done training model
generating training log
* BEGINNING FOLD 0
^CTraceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 363, in <module>
    model_train.train(dog, models[i], model_hparams[i], save_names[i], train_params[i])
  File "/home/jerry/work/earthlab/munge_data/models/model_train.py", line 44, in train
    model.train(dataset.train[i], dataset.validation[i])
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 110, in train
    npx, npy = self.aggregate(train_data)
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 101, in aggregate
    tx, ty = data[i]
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 486, in __getitem__
    ret_imgs[i] = self.load_dcube(self.path_prefix + self.X_ref[ret_indices[i]])
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 352, in load_dcube
    image_in = np.genfromtxt(cube_loc.replace("x_img", self.sreplace), delimiter=',')
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/numpy/lib/npyio.py", line 2396, in genfromtxt
    output = np.array(data, dtype)
KeyboardInterrupt

(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ^C
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ../create_data/
-bash: ../create_data/: Is a directory
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ls
custom_models   mutils.py   nohup.out    saved_models   train_1.py   train_frame.py
logger.py       no2hup.out  __pycache__  test2model.py  train_3.py   train_noise.py
model_train.py  no3hup.out  readme.md    test_model.py  trainer2.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ cd ../create_data/
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls
alt_datacube.py          create_data.sh               match_create_set.py   rfdata_loader.py
analyze_clipped.py       datacube_set.py              mcs_interpolation.py  testbed.py
build_train_val_test.py  data_nn_gedi                 nohup.out             test_vnoi.py
check_clip.py            dat_obj.py                   pixelstats_lm.py      test_xdata.py
code_match.py            dcube_dist_analysis.py       __pycache__           tif_merge_convert.py
corr_analysis.py         h5_sanitycheck.py            readme.md             voronator.py
create_2.py              interpolation_comparison.py  reset_raster_nd.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls -al
total 420
drwxrwxr-x  4 jerry jerry  4096 Aug 29 00:33 .
drwxrwxr-x 11 jerry jerry  4096 Jun  2 05:30 ..
-rw-rw-r--  1 jerry jerry 23796 Jun  2 08:34 alt_datacube.py
-rw-rw-r--  1 jerry jerry 14660 Sep 15  2022 analyze_clipped.py
-rw-rw-r--  1 jerry jerry  2350 Sep 14  2022 build_train_val_test.py
-rw-rw-r--  1 jerry jerry  1443 Sep 13  2022 check_clip.py
-rw-rw-r--  1 jerry jerry  6390 Sep 13  2022 code_match.py
-rw-rw-r--  1 jerry jerry  2951 Sep 13  2022 corr_analysis.py
-rw-rw-r--  1 jerry jerry 52631 Dec 13  2022 create_2.py
-rw-rw-r--  1 jerry jerry  1762 Jul 13  2022 create_data.sh
-rw-rw-r--  1 jerry jerry 23358 Oct 26  2022 datacube_set.py
drwxrwxr-x  4 jerry jerry  4096 Jun 21  2022 data_nn_gedi
-rw-rw-r--  1 jerry jerry  6838 Jun  2 08:45 dat_obj.py
-rw-rw-r--  1 jerry jerry  5140 Nov  4  2022 dcube_dist_analysis.py
-rw-rw-r--  1 jerry jerry  2195 Sep 13  2022 h5_sanitycheck.py
-rw-rw-r--  1 jerry jerry  2675 Jun  2 08:16 interpolation_comparison.py
-rw-rw-r--  1 jerry jerry 44202 Sep 19  2022 match_create_set.py
-rw-rw-r--  1 jerry jerry 40115 Aug 29 00:24 mcs_interpolation.py
-rw-------  1 jerry jerry 58289 Sep 11  2022 nohup.out
-rw-rw-r--  1 jerry jerry  1359 Sep 19  2022 pixelstats_lm.py
drwxrwxr-x  2 jerry jerry  4096 Jun  2 08:45 __pycache__
-rw-rw-r--  1 jerry jerry 55778 Oct 26  2022 readme.md
-rw-rw-r--  1 jerry jerry  1256 Sep 13  2022 reset_raster_nd.py
-rw-rw-r--  1 jerry jerry  1302 Sep 13  2022 rfdata_loader.py
-rw-rw-r--  1 jerry jerry   196 Sep 13  2022 testbed.py
-rw-rw-r--  1 jerry jerry  4056 Sep 13  2022 test_vnoi.py
-rw-rw-r--  1 jerry jerry  2430 Sep 13  2022 test_xdata.py
-rw-rw-r--  1 jerry jerry  4170 Sep 13  2022 tif_merge_convert.py
-rw-rw-r--  1 jerry jerry   549 Sep 13  2022 voronator.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim build_train_val_test.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ python build_train_val_test.py ../data/data_interpolated split1 1 0.2 0.3
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building test set
ntotal 101
test  21
building validation fold 0
fold  56 24
rm: cannot remove '../data/data_interpolated/fold_data/split1/meta.txt': No such file or directory
done
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ^C
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim mcs_interpolation.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim dat_obj.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim alt_datacube.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim trainer2.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ cd ../models/
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ vim trainer2.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building sets...
meta folds:  1
Traceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 62, in <module>
    dm1 = datacube_loader(dataset, folding, d_shuffle, d_batch, d_xref, d_yref, d_h5ref, d_meanstd, d_mmode, d_omode, d_cmode, d_h5mode, "m1_ximg")
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/dat_obj.py", line 45, in __init__
    channel_names = rfdata_loader.d1loader("../data/" + dataname +
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/rfdata_loader.py", line 43, in d1loader
    with open(fpath) as f:
FileNotFoundError: [Errno 2] No such file or directory: '../data/data_interpolated/meta/channel_names.txt'
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['slope_clipped'], ['aspct_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['slope_clipped'], ['aspct_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['slope_clipped'], ['aspct_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (14, 14, 4)
initializing datafold: train set 0
datacube dimensions:  (14, 14, 4)
initializing datafold: validation set 0
datacube dimensions:  (14, 14, 4)
sending rfr_t1 to be trained
generating training log
* BEGINNING FOLD 0
Traceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 361, in <module>
    model_train.train(dm1, models[i], model_hparams[i], save_names[i], train_params[i])
  File "/home/jerry/work/earthlab/munge_data/models/model_train.py", line 44, in train
    model.train(dataset.train[i], dataset.validation[i])
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 110, in train
    npx, npy = self.aggregate(train_data)
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 101, in aggregate
    tx, ty = data[i]
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 486, in __getitem__
    ret_imgs[i] = self.load_dcube(self.path_prefix + self.X_ref[ret_indices[i]])
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 357, in load_dcube
    image_in = image_in[:, self.keep_ids]
IndexError: index 2 is out of bounds for axis 1 with size 2
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ cd ../create_data/
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls
alt_datacube.py          code_match.py     datacube_set.py         h5_sanitycheck.py            nohup.out         reset_raster_nd.py  test_xdata.py
analyze_clipped.py       corr_analysis.py  data_nn_gedi            interpolation_comparison.py  pixelstats_lm.py  rfdata_loader.py    tif_merge_convert.py
build_train_val_test.py  create_2.py       dat_obj.py              match_create_set.py          __pycache__       testbed.py          voronator.py
check_clip.py            create_data.sh    dcube_dist_analysis.py  mcs_interpolation.py         readme.md         test_vnoi.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls
alt_datacube.py          code_match.py     datacube_set.py         h5_sanitycheck.py            nohup.out         reset_raster_nd.py  test_xdata.py
analyze_clipped.py       corr_analysis.py  data_nn_gedi            interpolation_comparison.py  pixelstats_lm.py  rfdata_loader.py    tif_merge_convert.py
build_train_val_test.py  create_2.py       dat_obj.py              match_create_set.py          __pycache__       testbed.py          voronator.py
check_clip.py            create_data.sh    dcube_dist_analysis.py  mcs_interpolation.py         readme.md         test_vnoi.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls -al
total 420
drwxrwxr-x  4 jerry jerry  4096 Aug 29 01:04 .
drwxrwxr-x 11 jerry jerry  4096 Jun  2 05:30 ..
-rw-rw-r--  1 jerry jerry 23796 Jun  2 08:34 alt_datacube.py
-rw-rw-r--  1 jerry jerry 14660 Sep 15  2022 analyze_clipped.py
-rw-rw-r--  1 jerry jerry  2350 Sep 14  2022 build_train_val_test.py
-rw-rw-r--  1 jerry jerry  1443 Sep 13  2022 check_clip.py
-rw-rw-r--  1 jerry jerry  6390 Sep 13  2022 code_match.py
-rw-rw-r--  1 jerry jerry  2951 Sep 13  2022 corr_analysis.py
-rw-rw-r--  1 jerry jerry 52631 Dec 13  2022 create_2.py
-rw-rw-r--  1 jerry jerry  1762 Jul 13  2022 create_data.sh
-rw-rw-r--  1 jerry jerry 23358 Oct 26  2022 datacube_set.py
drwxrwxr-x  4 jerry jerry  4096 Jun 21  2022 data_nn_gedi
-rw-rw-r--  1 jerry jerry  6838 Jun  2 08:45 dat_obj.py
-rw-rw-r--  1 jerry jerry  5140 Nov  4  2022 dcube_dist_analysis.py
-rw-rw-r--  1 jerry jerry  2195 Sep 13  2022 h5_sanitycheck.py
-rw-rw-r--  1 jerry jerry  2675 Jun  2 08:16 interpolation_comparison.py
-rw-rw-r--  1 jerry jerry 44202 Sep 19  2022 match_create_set.py
-rw-rw-r--  1 jerry jerry 40121 Aug 29 00:44 mcs_interpolation.py
-rw-------  1 jerry jerry 58289 Sep 11  2022 nohup.out
-rw-rw-r--  1 jerry jerry  1359 Sep 19  2022 pixelstats_lm.py
drwxrwxr-x  2 jerry jerry  4096 Jun  2 08:45 __pycache__
-rw-rw-r--  1 jerry jerry 55778 Oct 26  2022 readme.md
-rw-rw-r--  1 jerry jerry  1256 Sep 13  2022 reset_raster_nd.py
-rw-rw-r--  1 jerry jerry  1302 Sep 13  2022 rfdata_loader.py
-rw-rw-r--  1 jerry jerry   196 Sep 13  2022 testbed.py
-rw-rw-r--  1 jerry jerry  4056 Sep 13  2022 test_vnoi.py
-rw-rw-r--  1 jerry jerry  2430 Sep 13  2022 test_xdata.py
-rw-rw-r--  1 jerry jerry  4170 Sep 13  2022 tif_merge_convert.py
-rw-rw-r--  1 jerry jerry   549 Sep 13  2022 voronator.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim mcs_interpolation.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ cd ../models/
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ls
custom_models  model_train.py  no2hup.out  nohup.out    readme.md     test2model.py  train_1.py  trainer2.py     train_noise.py
logger.py      mutils.py       no3hup.out  __pycache__  saved_models  test_model.py  train_3.py  train_frame.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ls -al
total 296
drwxrwxr-x  5 jerry jerry   4096 Aug 29 01:06 .
drwxrwxr-x 11 jerry jerry   4096 Jun  2 05:30 ..
drwxrwxr-x  3 jerry jerry   4096 Jun  2 04:51 custom_models
-rw-rw-r--  1 jerry jerry   6869 Sep 13  2022 logger.py
-rw-rw-r--  1 jerry jerry   2644 Sep 13  2022 model_train.py
-rw-rw-r--  1 jerry jerry   4235 Sep 19  2022 mutils.py
-rw-rw-r--  1 jerry jerry  11222 Sep 18  2022 no2hup.out
-rw-rw-r--  1 jerry jerry  10777 Sep 18  2022 no3hup.out
-rw-------  1 jerry jerry 142486 Mar 11 01:58 nohup.out
drwxrwxr-x  2 jerry jerry   4096 Oct 31  2022 __pycache__
-rw-rw-r--  1 jerry jerry   3269 Sep 12  2022 readme.md
drwxrwxr-x 83 jerry jerry  12288 Aug 29 01:10 saved_models
-rw-rw-r--  1 jerry jerry      0 Jul 13  2022 test2model.py
-rw-rw-r--  1 jerry jerry   3508 Sep 13  2022 test_model.py
-rw-rw-r--  1 jerry jerry   8549 Sep 18  2022 train_1.py
-rw-rw-r--  1 jerry jerry   8037 Oct 26  2022 train_3.py
-rw-rw-r--  1 jerry jerry  18571 Aug 29 01:06 trainer2.py
-rw-rw-r--  1 jerry jerry  18684 Mar 10 15:13 train_frame.py
-rw-rw-r--  1 jerry jerry   9886 Sep 21  2022 train_noise.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ vim trainer2.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$
(neon) jerry@woolley:~/work/earthlab/munge_data/models$
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ls
custom_models  model_train.py  no2hup.out  nohup.out    readme.md     test2model.py  train_1.py  trainer2.py     train_noise.py
logger.py      mutils.py       no3hup.out  __pycache__  saved_models  test_model.py  train_3.py  train_frame.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ cd
(neon) jerry@woolley:~$ python trainer2.py
python: can't open file '/home/jerry/trainer2.py': [Errno 2] No such file or directory
(neon) jerry@woolley:~$ ls
anaconda3  Desktop  Documents  Downloads  earth-analytics  Music  Pictures  Public  snap  Templates  Videos  work
(neon) jerry@woolley:~$ cd work/earthlab/munge_data/
(neon) jerry@woolley:~/work/earthlab/munge_data$ ls
 create_data   data   experiments   figures   logs   models   raw_data   README.md   ssh_transfer.txt  ' .txt'
(neon) jerry@woolley:~/work/earthlab/munge_data$ cd models/
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ls
custom_models  model_train.py  no2hup.out  nohup.out    readme.md     test2model.py  train_1.py  trainer2.py     train_noise.py
logger.py      mutils.py       no3hup.out  __pycache__  saved_models  test_model.py  train_3.py  train_frame.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python train)1,
-bash: syntax error near unexpected token `)'
(neon) jerry@woolley:~/work/earthlab/munge_data/models$
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (14, 14, 66)
initializing datafold: train set 0
datacube dimensions:  (14, 14, 66)
initializing datafold: validation set 0
datacube dimensions:  (14, 14, 66)
sending rfr_t1 to be trained
generating training log
* BEGINNING FOLD 0
Traceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 362, in <module>
    model_train.train(dm1, models[i], model_hparams[i], save_names[i], train_params[i])
  File "/home/jerry/work/earthlab/munge_data/models/model_train.py", line 44, in train
    model.train(dataset.train[i], dataset.validation[i])
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 110, in train
    npx, npy = self.aggregate(train_data)
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 101, in aggregate
    tx, ty = data[i]
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 486, in __getitem__
    ret_imgs[i] = self.load_dcube(self.path_prefix + self.X_ref[ret_indices[i]])
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 357, in load_dcube
    image_in = image_in[:, self.keep_ids]
IndexError: index 2 is out of bounds for axis 1 with size 2
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"

^CTraceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 9, in <module>
    import model_train
  File "/home/jerry/work/earthlab/munge_data/models/model_train.py", line 14, in <module>
    import train_1
  File "/home/jerry/work/earthlab/munge_data/models/train_1.py", line 8, in <module>
    from tensorflow import keras
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/tensorflow/__init__.py", line 473, in <module>
    keras._load()
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/tensorflow/python/util/lazy_loader.py", line 41, in _load
    module = importlib.import_module(self.__name__)
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/__init__.py", line 24, in <module>
    from keras import models
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/models/__init__.py", line 18, in <module>
    from keras.engine.functional import Functional
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/engine/functional.py", line 31, in <module>
    from keras.engine import training as training_lib
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/engine/training.py", line 30, in <module>
    from keras.engine import compile_utils
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/engine/compile_utils.py", line 20, in <module>
    from keras import metrics as metrics_mod
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/metrics/__init__.py", line 33, in <module>
    from keras.metrics.metrics import MeanRelativeError
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/metrics/metrics.py", line 22, in <module>
    from keras import activations
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/activations.py", line 20, in <module>
    import keras.layers.activation as activation_layers
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/layers/__init__.py", line 27, in <module>
    from keras.engine.base_preprocessing_layer import PreprocessingLayer
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/engine/base_preprocessing_layer.py", line 19, in <module>
    from keras.engine import data_adapter
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/keras/engine/data_adapter.py", line 38, in <module>
    import pandas as pd  # pylint: disable=g-import-not-at-top
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/pandas/__init__.py", line 48, in <module>
    from pandas.core.api import (
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/pandas/core/api.py", line 48, in <module>
    from pandas.core.groupby import (
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/pandas/core/groupby/__init__.py", line 1, in <module>
    from pandas.core.groupby.generic import (
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/pandas/core/groupby/generic.py", line 70, in <module>
    from pandas.core.frame import DataFrame
  File "/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/pandas/core/frame.py", line 157, in <module>
    from pandas.core.generic import NDFrame
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 879, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1012, in get_code
  File "<frozen importlib._bootstrap_external>", line 672, in _compile_bytecode
KeyboardInterrupt
^C
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (14, 14, 66)
initializing datafold: train set 0
datacube dimensions:  (14, 14, 66)
initializing datafold: validation set 0
datacube dimensions:  (14, 14, 66)
sending rfr_t1 to be trained
generating training log
* BEGINNING FOLD 0
Traceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 362, in <module>
    model_train.train(dm1, models[i], model_hparams[i], save_names[i], train_params[i])
  File "/home/jerry/work/earthlab/munge_data/models/model_train.py", line 44, in train
    model.train(dataset.train[i], dataset.validation[i])
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 110, in train
    npx, npy = self.aggregate(train_data)
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 101, in aggregate
    tx, ty = data[i]
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 486, in __getitem__
    ret_imgs[i] = self.load_dcube(self.path_prefix + self.X_ref[ret_indices[i]])
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 357, in load_dcube
    image_in = image_in[:, self.keep_ids]
IndexError: index 2 is out of bounds for axis 1 with size 2
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ cd ../create_data/
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ python build_train_val_test.py ../data/data_interpolated split1 1 0.2 0.3
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building test set
ntotal 100001
test  20001
building validation fold 0
fold  56000 24000
done
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ cd ../models/
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ ls
custom_models  model_train.py  no2hup.out  nohup.out    readme.md     test2model.py  train_1.py  trainer2.py     train_noise.py
logger.py      mutils.py       no3hup.out  __pycache__  saved_models  test_model.py  train_3.py  train_frame.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ python trainer2.py
/home/jerry/anaconda3/envs/neon/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (4, 4, 2)
initializing datafold: train set 0
datacube dimensions:  (4, 4, 2)
initializing datafold: validation set 0
datacube dimensions:  (4, 4, 2)
building sets...
meta folds:  1
channel names:  [['srtm_clipped'], ['nlcd_clipped'], ['cover'], ['cover_z_0'], ['cover_z_1'], ['cover_z_2'], ['cover_z_3'], ['cover_z_4'], ['cover_z_5'], ['cover_z_6'], ['cover_z_7'], ['cover_z_8'], ['cover_z_9'], ['cover_z_10'], ['cover_z_11'], ['cover_z_12'], ['cover_z_13'], ['cover_z_14'], ['cover_z_15'], ['cover_z_16'], ['cover_z_17'], ['cover_z_18'], ['cover_z_19'], ['cover_z_20'], ['cover_z_21'], ['cover_z_22'], ['cover_z_23'], ['cover_z_24'], ['cover_z_25'], ['cover_z_26'], ['cover_z_27'], ['cover_z_28'], ['cover_z_29'], ['pavd_z_0'], ['pavd_z_1'], ['pavd_z_2'], ['pavd_z_3'], ['pavd_z_4'], ['pavd_z_5'], ['pavd_z_6'], ['pavd_z_7'], ['pavd_z_8'], ['pavd_z_9'], ['pavd_z_10'], ['pavd_z_11'], ['pavd_z_12'], ['pavd_z_13'], ['pavd_z_14'], ['pavd_z_15'], ['pavd_z_16'], ['pavd_z_17'], ['pavd_z_18'], ['pavd_z_19'], ['pavd_z_20'], ['pavd_z_21'], ['pavd_z_22'], ['pavd_z_23'], ['pavd_z_24'], ['pavd_z_25'], ['pavd_z_26'], ['pavd_z_27'], ['pavd_z_28'], ['pavd_z_29'], ['fhd_normal']]
initializing datafold: test set
datacube dimensions:  (14, 14, 66)
initializing datafold: train set 0
datacube dimensions:  (14, 14, 66)

initializing datafold: validation set 0
datacube dimensions:  (14, 14, 66)
sending rfr_t1 to be trained
generating training log
* BEGINNING FOLD 0
Traceback (most recent call last):
  File "/home/jerry/work/earthlab/munge_data/models/trainer2.py", line 362, in <module>
    model_train.train(dm1, models[i], model_hparams[i], save_names[i], train_params[i])
  File "/home/jerry/work/earthlab/munge_data/models/model_train.py", line 44, in train
    model.train(dataset.train[i], dataset.validation[i])
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 110, in train
    npx, npy = self.aggregate(train_data)
  File "/home/jerry/work/earthlab/munge_data/models/custom_models/rf_regress.py", line 101, in aggregate
    tx, ty = data[i]
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 486, in __getitem__
    ret_imgs[i] = self.load_dcube(self.path_prefix + self.X_ref[ret_indices[i]])
  File "/home/jerry/work/earthlab/munge_data/models/../create_data/alt_datacube.py", line 357, in load_dcube
    image_in = image_in[:, self.keep_ids]
IndexError: index 2 is out of bounds for axis 1 with size 2
(neon) jerry@woolley:~/work/earthlab/munge_data/models$
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ vim trainer2.py
(neon) jerry@woolley:~/work/earthlab/munge_data/models$ cd ../create_data/
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls
alt_datacube.py          code_match.py     datacube_set.py         h5_sanitycheck.py            nohup.out         reset_raster_nd.py  test_xdata.py
analyze_clipped.py       corr_analysis.py  data_nn_gedi            interpolation_comparison.py  pixelstats_lm.py  rfdata_loader.py    tif_merge_convert.py
build_train_val_test.py  create_2.py       dat_obj.py              match_create_set.py          __pycache__       testbed.py          voronator.py
check_clip.py            create_data.sh    dcube_dist_analysis.py  mcs_interpolation.py         readme.md         test_vnoi.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ ls -al
total 436
drwxrwxr-x  4 jerry jerry  4096 Aug 29 04:17 .
drwxrwxr-x 11 jerry jerry  4096 Jun  2 05:30 ..
-rw-rw-r--  1 jerry jerry 23796 Jun  2 08:34 alt_datacube.py
-rw-rw-r--  1 jerry jerry 14660 Sep 15  2022 analyze_clipped.py
-rw-rw-r--  1 jerry jerry  2350 Sep 14  2022 build_train_val_test.py
-rw-rw-r--  1 jerry jerry  1443 Sep 13  2022 check_clip.py
-rw-rw-r--  1 jerry jerry  6390 Sep 13  2022 code_match.py
-rw-rw-r--  1 jerry jerry  2951 Sep 13  2022 corr_analysis.py
-rw-rw-r--  1 jerry jerry 52631 Dec 13  2022 create_2.py
-rw-rw-r--  1 jerry jerry  1762 Jul 13  2022 create_data.sh
-rw-rw-r--  1 jerry jerry 23358 Oct 26  2022 datacube_set.py
drwxrwxr-x  4 jerry jerry  4096 Jun 21  2022 data_nn_gedi
-rw-rw-r--  1 jerry jerry  6838 Jun  2 08:45 dat_obj.py
-rw-r--r--  1 jerry jerry 16384 Aug 29 04:17 .dat_obj.py.swp
-rw-rw-r--  1 jerry jerry  5140 Nov  4  2022 dcube_dist_analysis.py
-rw-rw-r--  1 jerry jerry  2195 Sep 13  2022 h5_sanitycheck.py
-rw-rw-r--  1 jerry jerry  2675 Jun  2 08:16 interpolation_comparison.py
-rw-rw-r--  1 jerry jerry 44202 Sep 19  2022 match_create_set.py
-rw-rw-r--  1 jerry jerry 40109 Aug 29 01:13 mcs_interpolation.py
-rw-------  1 jerry jerry 58289 Sep 11  2022 nohup.out
-rw-rw-r--  1 jerry jerry  1359 Sep 19  2022 pixelstats_lm.py
drwxrwxr-x  2 jerry jerry  4096 Jun  2 08:45 __pycache__
-rw-rw-r--  1 jerry jerry 55778 Oct 26  2022 readme.md
-rw-rw-r--  1 jerry jerry  1256 Sep 13  2022 reset_raster_nd.py
-rw-rw-r--  1 jerry jerry  1302 Sep 13  2022 rfdata_loader.py
-rw-rw-r--  1 jerry jerry   196 Sep 13  2022 testbed.py
-rw-rw-r--  1 jerry jerry  4056 Sep 13  2022 test_vnoi.py
-rw-rw-r--  1 jerry jerry  2430 Sep 13  2022 test_xdata.py
-rw-rw-r--  1 jerry jerry  4170 Sep 13  2022 tif_merge_convert.py
-rw-rw-r--  1 jerry jerry   549 Sep 13  2022 voronator.py
(neon) jerry@woolley:~/work/earthlab/munge_data/create_data$ vim alt_datacube.py

396                                     image[i] = (image[i] - m_s[0][self.keep_ids[i]]) / m_s[1][self.keep_ids[i]]
397                         else:
398                             for i in range(image.shape[0]):
399                                 if m_s[1][self.keep_ids[i]] != 0:
400                                     image[i] = (image[i] - m_s[0][i]) / m_s[1][i]
401             return image
402
403     def get_n_samples(self):
404         return len(self.y)
405
406     # def predict_mode (self, pmode):
407
408     def set_drop(self, dmode):
409         self.drop_channels = dmode
410         self.keep_ids = self.drops_to_keeps()
411
412     def drops_to_keeps(self):
413         kis = []
414         if self.drop_channels == []:
415             return []
416         else:
417             for i in range(len(self.channel_names)):
418                 if self.channel_names[i] not in self.drop_channels:
419                     kis.append(i)
420         return kis
421
422     def keeps_to_drops(self):
423         dis = []
424         if len(self.keep_ids) == self.nchannels:
425             return []
426         else:
427             for i in range(len(self.channel_names)):
428                 if i in self.keep_ids:
429                     dis.append(self.channel_names[i])
430         return dis
431
432     def unshuffle(self):
433         self.indexes = np.arange(self.full_data.shape[0])
434
435     def set_keeps(self, keeps):
436         self.keep_ids = keeps
437
438     def set_drops(self, drops):
439         self.drop_channels = drops
440
441     def set_return(self, rmode):
442         # if rmode == "x" or rmode == "y" or rmode == "both":
443         self.return_format = rmode
444         # else:
445         #    print("cannot set return mode - invalid mode provided")
446         return self
447
448     def set_flatten(self, fmode):
449         self.flat_mode = fmode
450
451     def __len__(self):
452         return self.lenn
453
454     def getindices(self, idx):
455         return self.indexes[idx * self.batch_size: min(((idx + 1) * self.batch_size), self.full_data.shape[0])]
456
457     def __getitem__(self, idx):
458         # return picture data batch
459         ### TODO - account for dropped channels in return size
460         ret_indices = self.indexes[idx * self.batch_size: min(((idx + 1) * self.batch_size), self.full_data.shape[0])]
461         if self.mem_sensitive:
462             # print(ret_indices)
463             if self.flat_mode:
464                 if self.drop_channels == []:
465                     ret_imgs = np.zeros((len(ret_indices), self.dims[0] * self.dims[1] * self.dims[2]))
                                                                                                                                                                   428,1         90%
