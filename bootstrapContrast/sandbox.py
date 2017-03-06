from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, mannwhitneyu, norm
from collections import OrderedDict
from numpy.random import randint
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, FixedLocator, AutoLocator, FormatStrFormatter
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, rcdefaults
import sys
import seaborn.apionly as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def contrastplot_test(data, x, y, idx=None, 
    
    alpha=0.75, 
    axis_title_size=None,

    barWidth=5,

    contrastShareY=True,
    contrastEffectSizeLineStyle='solid',
    contrastEffectSizeLineColor='black',
    contrastYlim=None,
    contrastZeroLineStyle='solid', 
    contrastZeroLineColor='black', 

    effectSizeYLabel="Effect Size", 

    figsize=None, 
    floatContrast=True,
    floatSwarmSpacer=0.2,

    heightRatio=(1, 1),

    lineWidth=2,
    legend=True,

    paired=False,
    pal=None, 

    rawMarkerSize=8,
    rawMarkerType='o',
    reps=3000,
    
    showGroupCount=True,
    show95CI=False, 
    showAllYAxes=False,
    showRawData=True,
    smoothboot=False, 
    statfunction=None, 
    summaryBar=False, 
    summaryBarColor='grey',
    summaryColour='black', 
    summaryLine=True, 
    summaryLineStyle='solid', 
    summaryLineWidth=0.25, 
    summaryMarkerSize=10, 
    summaryMarkerType='o',
    swarmShareY=True, 
    swarmYlim=None, 

    tickAngle=45,
    tickAlignment='right',

    violinOffset=0.375,
    violinWidth=0.2, 

    xticksize=None,
    yticksize=None,

    **kwargs):

	    '''Takes a pandas dataframe and produces a contrast plot:
	    either a Cummings hub-and-spoke plot or a Gardner-Altman contrast plot.
	    -----------------------------------------------------------------------
	    Description of flags upcoming.'''

	    # Check that `data` is a pandas dataframe
	    if 'DataFrame' not in str(type(data)):
	        raise TypeError("The object passed to the command is not not a pandas DataFrame.\
	         Please convert it to a pandas DataFrame.")

	    	    # Get and set levels of data[x]    
	    if idx is None:
	        # No idx is given, so all groups are compared to the first one in the DataFrame column.
	        if paired:
	        	levs_tuple=(tuple(data[x].unique()), )
	        else:
	        	levs_tuple=tuple(np.unique(data[x])[0:2],)
	        widthratio=[1]
	        if len(data[x].unique()) > 2:
	            floatContrast=False
	    else:
	        # check if multi-plot or not
	        if all(isinstance(element, str) for element in idx):
	            # if idx is supplied but not a multiplot (ie single list or tuple) 
	            levs_tuple=(idx, )
	            widthratio=[1]
	            if len(idx) > 2:
	                floatContrast=False
	        elif all(isinstance(element, tuple) for element in idx):
	            # if idx is supplied, and it is a list/tuple of tuples or lists, we have a multiplot!
	            levs_tuple=idx
	            if (any(len(element) > 2 for element in levs_tuple) and floatContrast == True):
	                # if any of the tuples in idx has more than 2 groups, we turn set floatContrast as False.
	                floatContrast=False
	            # Make sure the widthratio of the seperate multiplot corresponds to how 
	            # many groups there are in each one.
	            widthratio=[]
	            for i in levs_tuple:
	                widthratio.append(len(i))
	    u=list()
	    for t in levs_tuple:
	        for i in np.unique(t):
	            u.append(i)
	    u=np.unique(u)

	    tempdat=data.copy()
	    # Make sure the 'x' column is a 'category' type.
	    tempdat[x]=tempdat[x].astype("category")
	    tempdat=tempdat[tempdat[x].isin(u)]

	    # Filters out values that were not specified in idx.
	    tempdat[x].cat.set_categories(u, ordered=True, inplace=True)

	    # Select only the columns for plotting and grouping. 
	    # Also set palette based on total number of categories in data['x'] or data['hue_column']
	    if 'hue' in kwargs:
	        data=data[ [x,y,kwargs['hue']] ]
	        u=kwargs['hue']
	    else:
	        data=data[[x,y]]
	        u=x
	    
	    # Drop all nans. 
	    data=data.dropna()

	    # Set clean style
	    sns.set(style='ticks')

	    # plot params
	    if axis_title_size is None:
	        axis_title_size=20
	    if yticksize is None:
	        yticksize=15
	    if xticksize is None:
	        xticksize=15

	    axisTitleParams={'labelsize' : axis_title_size}
	    xtickParams={'labelsize' : xticksize}
	    ytickParams={'labelsize' : yticksize}
	    svgParams={'fonttype' : 'none'}

	    rc('axes', **axisTitleParams)
	    rc('xtick', **xtickParams)
	    rc('ytick', **ytickParams)
	    rc('svg', **svgParams)


	    # initialise statfunction
	    if statfunction == None:
	        statfunction=np.mean

	    # Ensure summaryLine and summaryBar are not displayed together.
	    if summaryLine is True and summaryBar is True:
	        summaryBar=True
	        summaryLine=False
	    
	    # Here we define the palette on all the levels of the 'x' column.
	    # Thus, if the same pandas dataframe is re-used across different plots,
	    # the color identity of each group will be maintained.
	    if pal is None:
	        plotPal=dict( zip( data[u].unique(), sns.color_palette(n_colors=len(data[u].unique())) ) 
	                      )
	    else:
	        plotPal=pal

	    if swarmYlim is None:
	        swarm_ylim=np.array([np.min(tempdat[y]), np.max(tempdat[y])])
	    else:
	        swarm_ylim=np.array([swarmYlim[0],swarmYlim[1]])

	    if contrastYlim is not None:
	        contrastYlim=np.array([contrastYlim[0],contrastYlim[1]])

	    # Expand the ylim in both directions.
	    ## Find half of the range of swarm_ylim.
	    swarmrange=swarm_ylim[1] - swarm_ylim[0]
	    pad=0.1 * swarmrange
	    x2=np.array([swarm_ylim[0]-pad, swarm_ylim[1]+pad])
	    swarm_ylim=x2
	    
	    # Create list to collect all the contrast DataFrames generated.
	    contrastList=list()
	    contrastListNames=list()
	    
	    if figsize is None:
	        if len(levs_tuple) > 2:
	            figsize=(12,(12/np.sqrt(2)))
	        else:
	            figsize=(8,(8/np.sqrt(2)))

	    barWidth=barWidth/1000 # Not sure why have to reduce the barwidth by this much! 

	    if showRawData is True:
	        maxSwarmSpan=0.25
	    else:
	        maxSwarmSpan=barWidth         
	        
	    # Initialise figure, taking into account desired figsize.
	    fig=plt.figure(figsize=figsize)