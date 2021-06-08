# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 22:09:55 2021

@author: JZ2018
"""

def coalesce(df, column_names):
    i = iter(column_names)
    column_name = next(i)
    answer=df[column_name]
    for column_name in i:
            answer = answer.fillna(df[column_name])
    return answer

class renamer():
    def __init__(self):
        self.d = dict()
    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])
        
from math import sin, cos, sqrt, atan2, radians
def latlongdistKM(lat1, long1, lat2, long2):
    R=6371
    lat1_r, lat2_r, long1_r, long2_r = (radians(x) for x in [lat1, lat2, long1, long2])
    dlon = long2_r - long1_r
    dlat = lat2_r - lat1_r
    
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2) **2
    a=abs(a)

    c=2 * atan2(sqrt(a), sqrt(1-a))
    dist = R*c
    return dist

def dist_weight(x):
    
    minx = 0.1
    maxx = 0.9
    curv = 1
    startx = 10
    endx = 70
    pdc = minx + (maxx - minx) / (1+curv**(startx+endx/2-x)/endx)
    return 1-pdc
