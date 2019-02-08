# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
import pandas as pd
#import seaborn as sns

directory = str(sys.argv[1])

ins = pd.DataFrame()
for filename in os.listdir(directory):
    
    #output = outputdir+filename.replace(".matrix.gz", ".txt")
    filename=directory+filename
    #read contact matrix
    insulation = pd.read_csv(filename,\
                       index_col=0, header=0,comment="#")
    insulation=insulation[insulation.insulation<20]
    #sns.distplot(insulation.iloc[:,0],bins=5)
    ins=pd.concat([ins,insulation], axis=0)
    #tad = np.column_stack((tad_index,tad_boundary))
ins.to_csv(directory+"insulation.txt",index=True)
    #np.savetxt(output, tad, delimiter='\t')
    #sns.distplot(tad_boundary,bins=10)
    #sns.heatmap(cm_log[0:60,0:60])
