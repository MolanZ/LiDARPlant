"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
-----------------------------------------------------------------------
Copyright: 2018, Jan Zörner
Licence: GNU GPLv3
"""

from datetime import datetime

from pycrown import PyCrown
import sys

if __name__ == '__main__':

    TSTART = datetime.now()

    F_CHM = 'data/chm_m.tif'
    F_DTM = 'data/dtm_m.tif'
    F_DSM = 'data/dsm_m.tif'
    F_LAS = 'data/las.las'

    PC = PyCrown(F_CHM, F_DTM, F_DSM, F_LAS, outpath='result')
    # Cut off edges
    # PC.clip_data_to_bbox((1802200, 1802400, 5467250, 5467450))

    # Smooth CHM with 5m median filter
    PC.filter_chm(5, ws_in_pixels=True)


    PC.export_raster(PC.chm, PC.outpath / 'chm.tif', 'CHM')


    TEND = datetime.now()

    print(f'Processing time: {TEND-TSTART} [HH:MM:SS]')
