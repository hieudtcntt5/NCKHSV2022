import json
import cv2
import numpy as np


path = "D:/hieu_dt/Nam3_KI_2/NCKH/Br35H-Mask-RCNN/VAL/"

f = open("annotations_val.json")

data = json.load(f)

for i in data:
    filename = data[i]['filename']
    image = cv2.imread(path+str(filename))
    dimenssion = image.shape
    tmp = np.zeros(dimenssion).astype('uint8')
    point = data[i]['regions'][0]
    all_point =  point['shape_attributes']
    if len(all_point)==3:
        x = all_point['all_points_x']
        y = all_point['all_points_y']
        pts = []
        for j in range(0,len(x)):            
            pts.append([x[j],y[j]])
        pts = np.array(pts)
        pts = pts.reshape((-1,1,2))
        isClosed = True
        tmp = cv2.fillPoly(tmp,pts=[pts],color=(255,255,255))
    elif len(data[i]['regions'][0]['shape_attributes']) == 6:
        center_cooriinates = (data[i]['regions'][0]['shape_attributes']['cx'],
                                data[i]['regions'][0]['shape_attributes']['cy'])
        axesLength = (int(data[i]['regions'][0]['shape_attributes']['rx']),
                        int(data[i]['regions'][0]['shape_attributes']['ry']))
        angle = data[i]['regions'][0]['shape_attributes']['theta']
        startAngle = 0
        eniAngle = 360
        tmp = cv2.ellipse(tmp,
                            center_cooriinates,
                            axesLength,
                            angle,
                            startAngle,
                            eniAngle,
                            (255,255,255),
                            thickness=-1)
    filename = str(filename).replace('.jpg','')
    cv2.imwrite("{}.png".format(path + filename), tmp.astype('uint8'))  














    
