# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:56:30 2019

@author: Yang Xu

The method is adopted and modified from
https://www.sciencedirect.com/science/article/pii/S0092867414014974?via%3Dihub

"""

#import re
#import os
import scipy.ndimage.filters as filters
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
from scipy import signal
#from scipy import ndimage
#from scipy.ndimage import gaussian_filter

##read interaction contact matrix  
filename="Xray-GM12878-Control-R1__hg19__genome__C-100000-iced__chr1.matrix.gz"
filename="chr17_cm.score.matrix.gz"
hic = pd.read_table(filename,index_col=0, header=0,comment="#")
hic = hic.values
hic = np.nan_to_num(hic)
hic = np.log(hic+1)
hic = hic[:7782,:7782]

##filter and threshold
#f = np.array((-1,-1,-1,6,-1,-1,-1))

def filter2d(gk):
    n,m=gk.shape
    f = np.ones(gk.shape)
    f[n//2+1:,:m//2]=0
    f*=-1
    f[n//2,m//2]=abs(np.sum(f))-1
    #f=np.multiply(gk,f)
    return f

def filter2d_test(gk):
    n,m=gk.shape
    f = np.ones(gk.shape)
    f[n//2,m//2]=n*m-1
    #f=np.multiply(gk,f)
    return f

def filter1d(gk):
    n,m=gk.shape
    f = np.ones((m))
    f*=-1
    f[m//2]=m-1
    #f=np.multiply(gk,f)
    return f

##find local maximum
def find_dot(hic,f,thresh,dist):
    
    hic_x = np.zeros(hic.shape)
    hic_y = np.zeros(hic.shape)

    for i in range(hic.shape[0]):
        x=signal.convolve(hic[i,:],f, mode='same')
        y=signal.convolve(hic[:,i],f, mode='same')
        peak_x, _ = signal.find_peaks(x, distance=dist)
        peak_y, _ = signal.find_peaks(y, distance=dist)
        signal_x=np.zeros((hic.shape[0]))
        signal_x[peak_x]=1
        signal_x[hic[i,:]<thresh]=0
        
        signal_y=np.zeros((hic.shape[0]))
        signal_y[peak_y]=1
        signal_y[hic[:,i]<thresh]=0
        
        hic_x[i,:]=signal_x
        hic_y[:,i]=signal_y
    
    loop = hic_x*hic_y
    loop_loc = np.where(loop==1)
    loop_loc = np.column_stack((loop_loc[0],loop_loc[1]))
    
    return loop_loc

def find_maxdot(hic,f,thresh,dist,ratio=2):
    
    dhic = signal.convolve2d(hic, f, boundary='symm', mode='same')
    maxima = filters.maximum_filter(dhic,dist)
    maxima = (dhic == maxima)
    maxima = maxima.astype(np.int)
    nhic = np.multiply(hic,maxima)
    nhic[nhic<thresh]=0
    loop_loc = np.where(nhic!=0)
    loop_loc = np.column_stack((loop_loc[0],loop_loc[1]))
    
    return loop_loc

##distance threshold
def dist_clean(loop_loc,mind=30,maxd=100):
    
    new_loop_loc=[]
    for i in range(loop_loc.shape[0]):
        if loop_loc[i,1]-loop_loc[i,0]>=mind:
            if loop_loc[i,1]-loop_loc[i,0]<=maxd:
                new_loop_loc.append([loop_loc[i,0],loop_loc[i,1]])
    new_loop_loc=np.asarray(new_loop_loc)
    
    return new_loop_loc

##gaussian kernal
def gaussin_kernal(size=(29,29),sigma=7):
    x, y =size[0],size[1]
    x, y = np.mgrid[-(x//2):(x//2)+1, -(y//2):(y//2)+1]
    g = np.exp(-((x/sigma)**2+(y/sigma)**2)/2)
    return g/g.sum()

##rotate the image without cutting the image
def rotation(image,angle):
    
    (h, w) = image.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    ##new bounding of image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    ##adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    ##perform the actual rotation and return the image
    dst=cv2.warpAffine(image, M, (nW, nH))
    
    return dst

def dresponse(dhic,loop):
    res=[]
    for i in range(loop.shape[0]):
        res.append(dhic[loop[i,0],loop[i,1]])
    return res

###############################################################################
f = np.array((-1,-2,-1))
loop_loc=find_dot(hic,f,thresh=3,dist=3)
new_loop_loc=dist_clean(loop_loc)

#new_loop = np.zeros((hic.shape))
#for i in range(new_loop_loc.shape[0]):
#    new_loop[new_loop_loc[i,0],new_loop_loc[i,1]]=1
    
gk = gaussin_kernal(size=(3,3),sigma=1)
f = filter1d(gk)
hic_1=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_1 = find_dot(hic_1,f,thresh=3,dist=3)
new_loop_loc_1=dist_clean(loop_loc_1)

gk = gaussin_kernal(size=(5,5),sigma=2)
f = filter1d(gk)
hic_2=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_2 = find_dot(hic_2,f,thresh=3,dist=5)
new_loop_loc_2=dist_clean(loop_loc_2)

gk = gaussin_kernal(size=(7,7),sigma=3)
f = filter1d(gk)
hic_3=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_3 = find_dot(hic_3,f,thresh=1.5,dist=7)
new_loop_loc_3=dist_clean(loop_loc_3)

gk = gaussin_kernal(size=(11,11),sigma=5)
f = filter1d(gk)
hic_4=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_4 = find_dot(hic_4,f,thresh=3,dist=11)
new_loop_loc_4=dist_clean(loop_loc_4)

gk = gaussin_kernal(size=(15,15),sigma=7)
f = filter1d(gk)
hic_5=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_5 = find_dot(hic_5,f,thresh=3,dist=15)
new_loop_loc_5=dist_clean(loop_loc_5)

gk = gaussin_kernal(size=(19,19),sigma=9)
f = filter1d(gk)
hic_6=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_6 = find_dot(hic_6,f,thresh=3,dist=19)
new_loop_loc_6=dist_clean(loop_loc_6)

gk = gaussin_kernal(size=(23,23),sigma=11)
f = filter1d(gk)
hic_7=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_7 = find_dot(hic_7,f,thresh=1.5,dist=23)
new_loop_loc_7=dist_clean(loop_loc_7)

###############################################################################
f = np.array([(-1,-1,-1),(-1,8,-1),(-1,-1,-1)])
f[2,0]=0
loop_loc=find_maxdot(hic,f,thresh=3,dist=3)
new_loop_loc=dist_clean(loop_loc)

gk = gaussin_kernal(size=(3,3),sigma=1)
f = filter2d(gk)
hic_1=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_1 = find_maxdot(hic_1,f,thresh=3,dist=3)
new_loop_loc_1=dist_clean(loop_loc_1)

gk = gaussin_kernal(size=(5,5),sigma=2)
f = filter2d(gk)
hic_2=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_2 = find_maxdot(hic_2,f,thresh=3,dist=5)
new_loop_loc_2=dist_clean(loop_loc_2)

gk = gaussin_kernal(size=(5,5),sigma=2)
f = filter2d(gk)
hic_3=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_3 = find_maxdot(hic_3,f,thresh=2,dist=5)
new_loop_loc_3=dist_clean(loop_loc_3,mind=10,maxd=30)

gk = gaussin_kernal(size=(11,11),sigma=5)
f = filter2d(gk)
hic_4=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_4 = find_maxdot(hic_4,f,thresh=0.5,dist=11)
new_loop_loc_4=dist_clean(loop_loc_4,mind=25,maxd=150)


new_loop_loc_3=new_loop_loc_3[0:new_loop_loc_3.shape[0]-11]
ratio=[]
ft = filter2d_test(gk)
m,n = ft.shape
for i in range(new_loop_loc_3.shape[0]):
    x=new_loop_loc_3[i,0]
    y=new_loop_loc_3[i,1]
    a = ft[m//2,n//2]*hic_3[x,y]
    b = np.sum(np.multiply(ft,hic_3[x-2:x+3,y-2:y+3]))
    ratio.append(np.log2(a/(b-a)))
    
ratio=np.asarray(ratio)
new_loop_loc_3=new_loop_loc_3[ratio>=0.05]

gk = gaussin_kernal(size=(23,23),sigma=11)
f = filter2d(gk)
hic_4=signal.convolve2d(hic, gk, boundary='symm', mode='same')
loop_loc_4 = find_maxdot(hic_4,f,thresh=1.5,dist=23)
new_loop_loc_4=dist_clean(loop_loc_4)

###############################################################################
loop_loc = loop_loc.tolist()
loop_loc_1 = loop_loc_1.tolist()
match = list(set(loop_loc).intersection(set(loop_loc_1)))

d1hic=hic-hic_1
res1=dresponse(d1hic,new_loop_loc)
    
d2hic=hic_1-hic_2
res2=dresponse(d2hic,new_loop_loc)

d3hic=hic_2-hic_3
res3=dresponse(d3hic,new_loop_loc)

d4hic=hic_3-hic_4
res4=dresponse(d4hic,new_loop_loc)

res = np.asarray((res1,res2,res3,res4))
res_heat=sns.clustermap(res,cmap="RdBu",row_cluster=False)

new_loop_loc_2 = new_loop_loc[res[3,:]>0,:]
###############################################################################
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
new_loop=np.zeros((hic.shape))
for i in range(new_loop_loc_3.shape[0]):
    new_loop[new_loop_loc_3[i,0],new_loop_loc_3[i,1]]=1

new_loop*=10
heat=sns.heatmap((new_loop+hic)[500:650,500:650],cmap=cmap)
heat.get_figure().savefig("heat_loop.jpeg",dpi=3600)
#heat=sns.heatmap(d1hic[1000:1400,1000:1400],cmap="RdBu")

###############################################################################
##aggregate peaks
a = np.zeros((21,21))
for i in range(new_loop_loc_4.shape[0]):
    x=new_loop_loc_4[i,0]
    y=new_loop_loc_4[i,1]
    #patch=hic_3[x-35:x+36,y-35:y+36]
    #new_patch = rotation(patch,45)
    #(cX,cY) = (new_patch.shape[0]//2,new_patch.shape[1]//2)
    #new_patch = new_patch[cX-10:cX+11,cY-10:cY+11]
    
    #a+= new_patch
    a+=hic_3[x-10:x+11,y-10:y+11]

heat=sns.heatmap(np.log10(a),cmap="RdBu")

a = np.zeros((11,11))
for i in range(new_loop_loc_3.shape[0]):
    x=new_loop_loc_3[i,0]
    y=new_loop_loc_3[i,1]
    #patch=hic_3[x-35:x+36,y-35:y+36]
    #new_patch = rotation(patch,45)
    #(cX,cY) = (new_patch.shape[0]//2,new_patch.shape[1]//2)
    #new_patch = new_patch[cX-10:cX+11,cY-10:cY+11]
    
    #a+= new_patch
    a+=hic_3[x-5:x+6,y-5:y+6]

heat=sns.heatmap(np.log10(a),cmap=cmap)
heat.get_figure().savefig("aggregate_peaks.jpeg",dpi=3600)

new_patch = rotation(a,-45)
(cX,cY) = (new_patch.shape[0]//2,new_patch.shape[1]//2)
new_patch = new_patch[cX-10:cX+11,cY-10:cY+11]
    
heat=sns.heatmap(np.log(new_patch),cmap="RdBu")
