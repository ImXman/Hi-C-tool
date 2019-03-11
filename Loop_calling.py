#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:01:53 2019

@author: imxman
"""

#import re
#import os
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import signal
#from scipy import ndimage
#from scipy.ndimage import gaussian_filter
    
filename="/home/imxman/Xray/Xray-GM12878-Control-R1__hg19__txt\
/C-100000/iced/Xray-GM12878-Control-R1__hg19__genome__C-100000-iced__chr3.matrix.gz"
hic = pd.read_table(filename,\
                       index_col=0, header=0,comment="#")
hic = hic.values
hic = np.nan_to_num(hic)
thresh = 4

#strut = np.array([[0,1,0],[1,1,1],[0,1,0]])
f = np.array((-1,0,1))

hic = np.log(hic+1)
#blur = gaussian_filter(hic, sigma=3,order=1)
hic_x = np.zeros(hic.shape)
hic_y = np.zeros(hic.shape)

for i in range(hic.shape[0]):
    x=signal.convolve(hic[i,:],f, mode='same')
    y=signal.convolve(hic[:,i],f, mode='same')
    peak_x, _ = signal.find_peaks(x, distance=9)
    peak_y, _ = signal.find_peaks(y, distance=9)
    #peak_x = signal.argrelextrema(x, np.greater)
    #peak_y = signal.argrelextrema(y, np.greater)
    signal_x=np.zeros((hic.shape[0]))
    signal_x[peak_x]=1
    signal_x[hic[i,:]<=thresh]=0
    
    signal_y=np.zeros((hic.shape[0]))
    signal_y[peak_y]=1
    signal_y[hic[:,i]<=thresh]=0
    #signals=ndimage.binary_dilation(signals, structure=np.array((1,1,1))).\
    #astype(signals.dtype)
    signal_x=signal_x.reshape(1,len(signal_x))
    #signal_y=signal_y.reshape(len(signal_y),1)
    hic_x[i,:]=signal_x
    hic_y[:,i]=signal_y
    


loop = hic_x*hic_y
#loop = ndimage.binary_erosion(loop, structure=strut).astype(loop.dtype)
loop*=10
heat=sns.heatmap((loop+hic)[200:400,200:400],cmap="RdBu")
heat.get_figure().savefig("heat_loop.jpeg",dpi=1200)