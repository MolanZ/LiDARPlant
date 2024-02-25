#author CUIT_RS191_TCY
from osgeo import gdal,gdal_array
import os
import numpy as np
import shapefile
from PIL import Image,ImageDraw
import csv
import codecs

def write_tif(img_data,output_name,proj,geoinfo):
    driver = gdal.GetDriverByName('GTiff')
    cols = img_data.shape[2]
    rows = img_data.shape[1]
    out_file = driver.Create(output_name,cols,rows,9,6)
    out_file.SetGeoTransform(geoinfo)
    out_file.SetProjection(proj)
    for i in range(9):
        out_file.GetRasterBand(i+1).WriteArray(img_data[i,:,:])
    del out_file

def geo2pixel(geoTrans,g_x,g_y):
    x = (g_x - geoTrans[0])/geoTrans[1]
    y = (g_y - geoTrans[3])/geoTrans[5]
    return x,y

fn = 'Mobot_Trees_20210127_v2.csv'
file_name = 'code/pycrown-master/example/data/rgbd_f.tif'
data = gdal.Open(file_name)
data_Array_nrgbre = gdal_array.LoadFile(file_name)
file_name = 'code/pycrown-master/example/data/chm_f.tif'
data = gdal.Open(file_name)
data_Array_chm = gdal_array.LoadFile(file_name)
proj = data.GetProjection()
geoTrans = data.GetGeoTransform()
width = data.RasterXSize
height = data.RasterYSize
width
height
np.shape(data_Array_nrgbre)
data_Array = np.zeros((9,height,width))

data_Array[0,:,:] = data_Array_chm
del data_Array_chm
data_Array[1,:,:] = (data_Array_nrgbre[0,:,:]-data_Array_nrgbre[1,:,:])/(data_Array_nrgbre[0,:,:]+data_Array_nrgbre[1,:,:])
data_Array[2,:,:] = (data_Array_nrgbre[0,:,:]-data_Array_nrgbre[4,:,:])/(data_Array_nrgbre[0,:,:]+data_Array_nrgbre[4,:,:])
data_Array[3,:,:] = data_Array_nrgbre[1,:,:] -data_Array_nrgbre[3,:,:]
data_Array[4,:,:] = data_Array_nrgbre[0,:,:]
data_Array[5,:,:] = data_Array_nrgbre[1,:,:]
data_Array[6,:,:] = data_Array_nrgbre[2,:,:]
data_Array[7,:,:] = data_Array_nrgbre[3,:,:]
data_Array[8,:,:] = data_Array_nrgbre[4,:,:]
del data_Array_nrgbre
i = 0

with codecs.open(fn, 'r','utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # skip the header
    next(reader,'None')
    #loop through each of the rows and assign the attributes to variables
    for row in reader:
        if row[7] == '<Null>' or float(row[7]) == 0.:
            r = 5/0.07
        else:
            r = float(row[7])/2/0.07
        output_file = 'data/08_23/'+str(row[11])
        if not os.path.exists(output_file):
            os.mkdir(output_file)
        output_file = 'data/08_23/'+str(row[11])+'/'+str(row[12])
        if not os.path.exists(output_file):
            os.mkdir(output_file)
        output_file = 'data/08_23/'+str(row[11])+'/'+str(row[12])+'/'+str(row[13])
        if not os.path.exists(output_file):
            os.mkdir(output_file)
        output_name = output_file + '/'+str(i)+'.tif'
        center_x,center_y = geo2pixel(geoTrans, float(row[18]), float(row[17]))
        if center_x*center_y <= 0 or center_x>=width or center_y>=height:
            continue
        start_x = int(max(center_x - r,0))
        start_y = int(max(center_y - r,0))
        end_x = int(min(center_x + r, width))
        end_y = int(min(center_y + r, height))
        clip = data_Array[:,start_y:end_y,start_x:end_x]
        geoinfo_new = list(geoTrans)
        geoinfo_new[0] = geoTrans[0]+geoTrans[1]*start_x;
        geoinfo_new[3] = geoTrans[3]+geoTrans[5]*start_y;
        write_tif(clip,output_name,proj,geoinfo_new)
        i += 1
i
