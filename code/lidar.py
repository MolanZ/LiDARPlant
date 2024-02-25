import sys
import lidargc.createDB as db
import lidargc.classify as chm



sys.path.append('code/')
sys.path


db.run()

import laspy
from laspy.file import File
las = laspy.read('../data/Mobot_November.las')
las.header.version
indata = laspy.open('../data/Mobot_November.las','r')
filename = infile.split("/")[-1]
points = indata.points
points[0].intensity
list(las.point_format.dimension_names)
points
for i, elem in enumerate(points):
    print(len(elem))
    X,Y,Z,intensity,flag_byte,raw_classification,scan_angle_rank,user_data,pt_src_id,gps_time=elem[0]
