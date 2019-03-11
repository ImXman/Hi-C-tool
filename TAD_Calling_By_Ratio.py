# -*- coding: utf-8 -*-
"""
Yang Xu

The method is adopted from 
https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-3387-6
"""
import os
import sys
import numpy as np
import pandas as pd
#import seaborn as sns
from scipy.signal import find_peaks
#from scipy.signal import argrelextrema

directory = str(sys.argv[1])
outputdir = str(sys.argv[2])
bin_size = int(sys.argv[3])

for filename in os.listdir(directory):
    
    output = outputdir+filename.replace(".matrix.gz", ".txt")
    filename=directory+filename
    #read contact matrix
    cm = pd.read_table(filename,sep="\t",\
                       index_col=0, header=0,compression='gzip',comment="#")
    insulation_index=list(cm.index)
    cm = cm.values
    cm[np.isnan(cm)]=0
    #cm_log=cm.copy()
    #cm_log=np.log(cm_log+1)
    
    ##calculate ratio insulation score
    n,m = cm.shape
    #insulation_index=[]
    insulation_score=[]
    for i in range(n-1): 
        if (i+1) <= bin_size :
            l_tad = np.triu(cm[:i+1,:i+1]).mean()
            r_tad = np.triu(cm[i+1:i+1+bin_size,i+1:i+1+bin_size]).mean()
            inter = cm[:i+1,i+1:i+1+bin_size].mean()
            ratio = (max(l_tad,r_tad)+1)/(inter+1)
            #insulation_index.append(i)
            insulation_score.append(ratio)
        elif (i+bin_size)>=n:
            l_tad = np.triu(cm[i+1-bin_size:i+1,i+1-bin_size:i+1]).mean()
            r_tad = np.triu(cm[i+1:,i+1:]).mean()
            inter = cm[i+1-bin_size:i+1,i+1:].mean()
            ratio = (max(l_tad,r_tad)+1)/(inter+1)
            #insulation_index.append(i)
            insulation_score.append(ratio)
        else:
            l_tad = np.triu(cm[i+1-bin_size:i+1,i+1-bin_size:i+1]).mean()
            r_tad = np.triu(cm[i+1:i+1+bin_size,i+1:i+1+bin_size]).mean()
            inter = np.tril(cm[i+1-bin_size:i+1,i+1:i+1+bin_size]).mean()
            ratio = (max(l_tad,r_tad)+1)/(inter+1)
            #insulation_index.append(i)
            insulation_score.append(ratio)
    insulation_score=np.array(insulation_score).reshape(len(insulation_score))
    insulation_index=np.array(insulation_index).reshape(len(insulation_index))
    #signal=argrelextrema(insulation_score, np.greater)[0]
    peaks, _ = find_peaks(insulation_score, distance=bin_size)
    peaks=peaks.tolist()
    #tad_boundary=insulation_score[signal]
    tad_boundary=insulation_score[peaks]
    tad_index=insulation_index[peaks]
    tad=pd.DataFrame({'bin':tad_index,'insulation':tad_boundary})
    #tad = np.column_stack((tad_index,tad_boundary))
    tad.to_csv(output,index=False)
    #np.savetxt(output, tad, delimiter='\t')
    #sns.distplot(tad_boundary,bins=10)
    #sns.heatmap(cm_log[0:60,0:60])
