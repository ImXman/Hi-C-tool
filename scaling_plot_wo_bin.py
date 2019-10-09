#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:31:55 2019

@author: imxman
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cooltools.lib import numutils

dmin=int(10)
dmax=int(250000)
n_bins=50
distbins = numutils.logbins(dmin, dmax, N=n_bins)

chr1 = pd.read_csv("chr1.txt",header=None,sep="\t")
dist = np.abs(chr1.iloc[:,0].values- chr1.iloc[:,1].values)
dist = dist[dist<=dmax]
dist = dist[dist>=dmin]
dist_log = np.log10(dist)

_ = np.histogram(dist_log,bins=distbins)

plt.hist(dist_log, bins = np.log10(distbins))
sns.distplot(dist_log,bins = distbins, hist=True, kde=True)