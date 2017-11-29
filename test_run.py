#! /usr/bin/env python

# Load Libraries
import matplotlib as mpl
mpl.use('SVG')
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set(style='ticks',context='talk')
import bootstrap_contrast as bsc

import pandas as pd
import numpy as np
import scipy as sp

# Dummy dataset
dataset=list()
for seed in [10,11,12,13,14,15]:
    np.random.seed(seed) # fix the seed so we get the same numbers each time.
    dataset.append(np.random.randn(40))
df=pd.DataFrame(dataset).T
cols=['Control','Group1','Group2','Group3','Group4','Group5']
df.columns=cols
# Create some upwards/downwards shifts.
df['Group2']=df['Group2']-0.1
df['Group3']=df['Group3']+0.2
df['Group4']=(df['Group4']*1.1)+4
df['Group5']=(df['Group5']*1.1)-1
# Add gender column.
df['Gender']=np.concatenate([np.repeat('Male',20),np.repeat('Female',20)])

# bsc.__version__

f,c=bsc.contrastplot(data=df,
                     idx=(('Group1','Group3','Group2'),
                          ('Control','Group4')),
                     color_col='Gender',
                     custom_palette={'Male':'blue',
                                     'Female':'red'},
                     float_contrast=True,
                     swarm_label='my swarm',
                     contrast_label='contrast',
                     show_means='bars',
                     means_width=0.5,
                     show_std=True,
                     fig_size=(10,8))

f.savefig('testfig.svg',format='svg')
