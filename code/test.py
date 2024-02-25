import pandas as pd
from PIL import Image
import matplotlib.image as mpimg
from osgeo import ogr
import subprocess
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import pathlib
import os
import random
import shapefile
from osgeo import gdal,gdal_array
import csv
import codecs
import pickle
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
from geopandas import *
import csv
import pickle

def comman(data,spd,spd_pmfgrd,dsm,dtm,chm):
    dsm_tif = dsm.replace('.img','.tif')
    dsm_1_tif = dsm.replace('.img','_1.tif')
    dsm_m = dsm.replace('.img','_m.tif')
    dtm_tif = dtm.replace('.img','.tif')
    dtm_1_tif = dtm.replace('.img','_1.tif')
    dtm_m = dtm.replace('.img','_m.tif')
    x = 'spdtranslate --input '+data + ' --if LAS --of SPD -b 1 -x LAST_RETURN --keeptemp --output '+spd+' --input_proj 102696.wkt'
    x = x.split(' ')
    subprocess.call(x)
    x = 'spdpmfgrd -i '+spd+' -o '+spd_pmfgrd+' -b 0.5 -r 50 --initelev 0.1'
    x = x.split(' ')
    subprocess.call(x)
    x = 'spdinterp --dsm --topo -f HFA -b 1 -r 50 --in NATURAL_NEIGHBOR_CGAL -i '+spd_pmfgrd+' -o '+dsm
    x = x.split(' ')
    subprocess.call(x)
    x = 'spdinterp --dtm --topo -f HFA -b 1 -r 50 --in NATURAL_NEIGHBOR_CGAL -i '+spd_pmfgrd+' -o '+dtm
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdal_translate -of GTiff -a_nodata -1 '+dsm+' '+dsm_tif
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdal_translate -of GTiff -a_nodata -1 '+dtm+' '+dtm_tif
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdalwarp -s_srs EPSG:32615 -t_srs EPSG:32615 -srcnodata INT_MAX -dstnodata -1 '+dsm_tif+' '+dsm_1_tif
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdalwarp -s_srs EPSG:32615 -t_srs EPSG:32615 -srcnodata INT_MAX -dstnodata -1 '+dtm_tif+' '+dtm_1_tif
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdal_calc.py -A '+dsm_1_tif+' --outfile='+dsm_m+' --calc=A*0.3048 --NoDataValue=-1'
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdal_calc.py -A '+dtm_1_tif+' --outfile='+dtm_m+' --calc=A*0.3048 --NoDataValue=-1'
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdal_calc.py -A '+dsm_m+' -B '+dtm_m+' --outfile='+chm+' --calc=A-B --NoDataValue=-1'
    x = x.split(' ')
    subprocess.call(x)

def merge(outpath,folder):
    chm = outpath+'/chm.tif'
    chm0 = '/'.join(folder)+'/chm_0.tif'
    chm_f = '/'.join(folder)+'/chm_f.tif'
    temp = listdir('/'.join(folder),[])
    for i in temp:
        if '.img' in i and '_nrg' in i and '.xml' not in i:
            rgbd = i
            break
    rgbd0 = '/'.join(folder)+'/rgbd_0.tif'
    rgbd_f = '/'.join(folder)+'/rgbd_f.tif'
    x = 'gdalwarp -s_srs EPSG:26915 -t_srs EPSG:26915 -of GTiff -tr 0.05 0.05 -srcnodata INT_MAX -dstnodata -1 '+rgbd+' '+rgbd_f
    x = x.split(' ')
    subprocess.call(x)
    x = 'gdalwarp -s_srs EPSG:26915 -t_srs EPSG:26915 -of GTiff -tr 0.05 0.05 -srcnodata INT_MAX -dstnodata -1 '+chm+' '+chm_f
    x = x.split(' ')
    subprocess.call(x)

def write_tif(band,img_data,output_name,proj,geoinfo):
    driver = gdal.GetDriverByName('GTiff')
    cols = img_data.shape[2]
    rows = img_data.shape[1]
    out_file = driver.Create(output_name,cols,rows,band,6)
    out_file.SetGeoTransform(geoinfo)
    out_file.SetProjection(proj)
    for i in range(band):
        out_file.GetRasterBand(i+1).WriteArray(img_data[i,:,:])
    del out_file

def geo2pixel(geoTrans,g_x,g_y):
    x = (g_x - geoTrans[0])/geoTrans[1]
    y = (g_y - geoTrans[3])/geoTrans[5]
    return int(x),int(y)

def extract_single_tree_6(folder):
    folder = '/'.join(folder)
    fn = folder+'/result/tree_crown_poly_raster.shp'
    file_name = folder+'/rgbd_f.tif'
    data = gdal.Open(file_name)
    data_Array_nrgbre = gdal_array.LoadFile(file_name)
    width_r = data.RasterXSize
    height_r = data.RasterYSize
    file_name = folder+'/chm_f.tif'
    data = gdal.Open(file_name)
    data_Array_chm = gdal_array.LoadFile(file_name)
    proj = data.GetProjection()
    geoTrans = data.GetGeoTransform()
    width_c = data.RasterXSize
    height_c = data.RasterYSize
    width = min(width_c,width_r)
    height = min(height_c,height_r)
    data_Array = np.zeros((9,height,width))
    data_Array_nrgbre = np.resize(data_Array_nrgbre,(5,height,width))
    data_Array_chm = np.resize(data_Array_chm,(height,width))
    data_Array[0,:,:] = data_Array_chm
    del data_Array_chm
    data_Array[1,:,:] = (data_Array_nrgbre[0,:,:]-data_Array_nrgbre[1,:,:])/(data_Array_nrgbre[0,:,:]+data_Array_nrgbre[1,:,:])
    data_Array[2,:,:] = (data_Array_nrgbre[0,:,:]-data_Array_nrgbre[4,:,:])/(data_Array_nrgbre[0,:,:]+data_Array_nrgbre[4,:,:])
    data_Array[3,:,:] =  data_Array_nrgbre[1,:,:] -data_Array_nrgbre[3,:,:]
    data_Array[4:,:,:] = data_Array_nrgbre
    del data_Array_nrgbre

    i = 0
    os.mkdir(folder+'/single_tree')
    shp = shapefile.Reader(fn)
    reader = shp.shapes()
    for poly in reader:
        #poly = poly.to_crs("EPSG:4326")
        minX,minY,maxX,maxY = poly.bbox
        center_x = (maxX-minX)/2+minX
        center_y = (minY-maxY)/2+maxY
        loc =str(center_x)+'__'+str(center_y)+'__'+str(i)
        output_file = folder+'/single_tree/'
        output_name = output_file + loc+'.tif'
        start_pos_x,start_pos_y = geo2pixel(geoTrans, minX,maxY)
        end_pos_x,end_pos_y = geo2pixel(geoTrans, maxX,minY)

        if start_pos_x==end_pos_x or start_pos_y==end_pos_y:
            clip = data_Array
        else:
            clip = data_Array[:,start_pos_y:end_pos_y,start_pos_x:end_pos_x]
        geoinfo_new = list(geoTrans)
        geoinfo_new[0] = minX;
        geoinfo_new[3] = maxY;
        write_tif(9, clip,output_name,proj,geoinfo_new)
        i += 1

def extract_single_tree_4(folder):
    folder = '/'.join(folder)
    fn = folder+'/result/tree_crown_poly_raster.shp'
    file_name = folder+'/rgbd_f.tif'
    data = gdal.Open(file_name)
    data_Array_nrgbre = gdal_array.LoadFile(file_name)
    width_r = data.RasterXSize
    height_r = data.RasterYSize
    file_name = folder+'/chm_f.tif'
    data = gdal.Open(file_name)
    data_Array_chm = gdal_array.LoadFile(file_name)
    proj = data.GetProjection()
    geoTrans = data.GetGeoTransform()
    width_c = data.RasterXSize
    height_c = data.RasterYSize
    width = min(width_c,width_r)
    height = min(height_c,height_r)
    data_Array = np.zeros((7,height,width))
    data_Array_nrgbre = np.resize(data_Array_nrgbre,(4,height,width))
    data_Array_chm = np.resize(data_Array_chm,(height,width))
    data_Array[0,:,:] = data_Array_chm
    del data_Array_chm
    data_Array[1,:,:] = (data_Array_nrgbre[0,:,:]-data_Array_nrgbre[1,:,:])/(data_Array_nrgbre[0,:,:]+data_Array_nrgbre[1,:,:])
    data_Array[2,:,:] = (data_Array_nrgbre[0,:,:]-data_Array_nrgbre[3,:,:])/(data_Array_nrgbre[0,:,:]+data_Array_nrgbre[3,:,:])
    data_Array[3:,:,:] = data_Array_nrgbre
    del data_Array_nrgbre

    i = 0
    os.mkdir(folder+'/single_tree')
    shp = shapefile.Reader(fn)
    reader = shp.shapes()
    for poly in reader:
        #poly = poly.to_crs("EPSG:4326")
        minX,minY,maxX,maxY = poly.bbox
        center_x = (maxX-minX)/2+minX
        center_y = (minY-maxY)/2+maxY
        loc =str(center_x)+'__'+str(center_y)+'__'+str(i)
        output_file = folder+'/single_tree/'
        output_name = output_file + loc+'.tif'
        start_pos_x,start_pos_y = geo2pixel(geoTrans, minX,maxY)
        end_pos_x,end_pos_y = geo2pixel(geoTrans, maxX,minY)

        if start_pos_x==end_pos_x or start_pos_y==end_pos_y:
            clip = data_Array
        else:
            clip = data_Array[:,start_pos_y:end_pos_y,start_pos_x:end_pos_x]
        geoinfo_new = list(geoTrans)
        geoinfo_new[0] = minX;
        geoinfo_new[3] = maxY;
        write_tif(7, clip,output_name,proj,geoinfo_new)
        i += 1

def get_allfile(path):
    all_file = []
    for f in os.listdir(path):
        f_name = os.path.join(path, f)
        all_file.append(f_name)
    return all_file

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
            if len(os.listdir(path)) == 1:
                list_name.append(file_path)
    return list_name

def read_img(dataset_path):
    dataset = gdal.Open(dataset_path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    del dataset
    temp = np.where(np.isnan(im_data),-1,im_data)
    temp = np.transpose(temp,(1,2,0))
    final = np.resize(temp,(100,100,9))
    final = final.flatten()
    return final

def load_data_test(line):
    image = read_img(line)
    image =np.reshape(image,(1,-1))
    loc = line.split('/')[-1]
    loc = loc.split('__')
    lat = float(loc[0])
    lon = float(loc[1])
    return image,lat,lon

def pred(x,level,flag):
    filename = 'model/svm'+flag+level+'.sav'
    net = joblib.load(filename)
    filename = 'model/pca'+flag+level+'.sav'
    pcamodel = joblib.load(filename)
    data = pcamodel.transform(x)
    predicted=net.predict(data)
    return predicted

def csv_filter():
    df = pd.read_csv('results.csv')
    df['x_y'] = df['Latitude']*1e6+df['Longitude']
    count = df['x_y'].value_counts()
    output = pd.DataFrame([],columns=['Family','Genus','Species','Latitude','Longitude','CRS'])
    for i in range(len(count)):
        loc = count.index[i]
        df1 = df[df['x_y']==loc]
        f = list(df1['Family'])
        g = list(df1['Genus'])
        s = list(df1['Species'])
        ss = Counter(s)
        ss = ss.most_common()

        if len(ss)>1 and (ss[0][1] == ss[1][1]):
            gg = Counter(g)
            gg = gg.most_common()
            if len(gg)>1 and (gg[0][1] == gg[1][1]):
                ff = Counter(f)
                ff = ff.most_common()
                x = df1.loc[df1[df1['Family']==ff[0][0]].index]
                output=output.append(x.iloc[0,:-1],ignore_index=True,sort=True)
            else:
                x = df1.loc[df1[df1['Genus']==gg[0][0]].index]
                output=output.append(x.iloc[0,:-1],ignore_index=True,sort=True)
        else:
            x = df1.loc[df1[df1['Species']==ss[0][0]].index]
            output=output.append(x.iloc[0,:-1],ignore_index=True,sort=True)

    #output = output.drop(columns=['x_y'])
    output.to_csv('results0.csv')

def test():
    list_file = listdir('testdata',[])
    for data in list_file:
        if '.las' not in data:
            continue
        spd = data.replace('.las','.spd')
        spd_pmfgrd = data.replace('.las','_pmfgrd.spd')
        dsm = data.replace('.las','_dsm.img')
        dtm = data.replace('.las','_dtm.img')
        chm = data.replace('.las','_chm_m.tif')
        comman(data,spd,spd_pmfgrd,dsm,dtm,chm)
        dtm_m = dtm.replace('.img','_m.tif')
        dsm_m = dsm.replace('.img','_m.tif')
        folder = data.split('/')
        folder = folder[:-1]
        outpath = folder
        outpath.append('result')
        outpath = '/'.join(outpath)
        os.mkdir(outpath)
        a=subprocess.call(['python','code/pycrownmaster/example/example.py',chm, dtm_m, dsm_m, data,outpath])
        if a == 1:
            continue
        folder = folder[:-1]
        merge(outpath, folder)
        extract_single_tree_6(folder)

    results = open('results_temp.csv','w',newline='')
    csv_writer = csv.writer(results)
    csv_writer.writerow(['Family','Genus','Species','Latitude','Longitude','CRS'])
    with open('dict/family_dict.pkl','rb') as f:
        family_dict = pickle.load(f)
    with open('dict/genus_dict.pkl','rb') as f:
        genus_dict = pickle.load(f)
    with open('dict/spices_dict.pkl','rb') as f:
        spices_dict = pickle.load(f)

    file_list = listdir('testdata',[])
    for line in file_list:
        if '/single_tree/' not in line:
            continue
        x,lat,lon =load_data_test(line)
        flag = '_6_08_'
        pred_f = pred(x,'f',flag)
        f_name = family_dict[str(pred_f[0])]
        pred_g = pred(x,f_name,flag)
        g_name = genus_dict[str(pred_g[0])]
        pred_s = pred(x,g_name,flag)
        s_name = spices_dict[str(pred_s[0])]
        csv_writer.writerow([f_name,g_name,s_name,str(lat),str(lon),'EPSG:32615'])
        flag = '_6_11_'
        pred_f = pred(x,'f',flag)
        f_name = family_dict[str(pred_f[0])]
        pred_g = pred(x,f_name,flag)
        g_name = genus_dict[str(pred_g[0])]
        pred_s = pred(x,g_name,flag)
        s_name = spices_dict[str(pred_s[0])]
        csv_writer.writerow([f_name,g_name,s_name,str(lat),str(lon),'EPSG:32615'])
    results.close()
    csv_filter()

test()
