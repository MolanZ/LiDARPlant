#-*-coding:utf-8-*-
import shapefile as shp
import csv
import codecs
import os

def trans_point(folder, fn, idlng, idlat, delimiter=','):
    # create a point shapefile
    output_shp = shp.Writer("%s.shp"%fn.split('.')[0])
    # for every record there must be a corresponding geometry.
    output_shp.autoBalance = 1
    # create the field names and data type for each.you can omit fields here
    # output_shp.field('id','N') # number
    output_shp.field('longitude', 'F', 10, 8) # float
    output_shp.field('latitude', 'F', 10, 8) # float
    output_shp.field('family','C',100) # string, max-length
    output_shp.field('genus','C',100)
    output_shp.field('species','C',100)
    # access the CSV file
    with codecs.open(fn, 'rb', 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # skip the header
        next(reader, None)
        #loop through each of the rows and assign the attributes to variables
        for row in reader:

            # idx = row[0]
            family= row[2]
            genus= row[3]
            species= row[6]
            lng= float(row[4])
            lat = float(row[5])
            # create the point geometry
            output_shp.point(lng, lat)
            # add attribute data
            output_shp.record(lng, lat, family, genus, species)
    output_shp.close() # save the Shapefile

if __name__ == '__main__':
    folder = '/'
    fn = 'results/results.csv'
    trans_point(folder, fn, 18, 19)
