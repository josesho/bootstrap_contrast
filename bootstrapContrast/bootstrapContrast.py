from scipy.stats import ttest_ind, mannwhitneyu, norm
from collections import OrderedDict
from numpy.random import randint
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from decimal import Decimal
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import warnings

# Taken without modification from scikits.bootstrap package
# Keep python 2/3 compatibility, without using six. At some point,
# we may need to add six as a requirement, but right now we can avoid it.
try:
    xrange
except NameError:
    xrange = range
    
def bootstrap_indexes(data, n_samples=5000):
    # Taken without modification from scikits.bootstrap package
    """
From the scikits.bootstrap package.
Given data points data, where axis 0 is considered to delineate points, return
an generator for sets of bootstrap indexes. This can be used as a list
of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
    """
    for _ in xrange(n_samples):
        yield randint(data.shape[0], size=(data.shape[0],))

def jackknife_indexes(data):
    # Taken without modification from scikits.bootstrap package
    """
From the scikits.bootstrap package.
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.
For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.
    """
    base = np.arange(0,len(data))
    return (np.delete(base,i) for i in base)

def getstatarray(tdata, statfunction, reps, sort = True):
    # Convenience function for use within `bootstrap` and `bootstrap_contrast`.
    # Produces `reps` number of bootstrapped samples for `tdata`, using `statfunction`
    # We don't need to generate actual samples that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes = bootstrap_indexes( tdata[0], reps ) # I use the scikits.bootstrap function here.
    statarray = np.array([statfunction(*(x[indexes] for x in tdata)) for indexes in bootindexes])
    if sort is True:
        statarray.sort(axis=0)
    return statarray

def checkInt(l):
    # convenience command to check column type
    # Not really used anymore but good code to just keep around?
    for item in l:
        if isinstance(item, int) is False:
            return False
        else:
            return True
        
def bca(data, alphas, statarray, statfunction, ostat, reps):
    # Subroutine called to calculate the BCa statistics

    # The bias correction value.
    z0 = norm.ppf( ( 1.0*np.sum(statarray < ostat, axis=0)  ) / reps )

    # Statistics of the jackknife distribution
    jackindexes = jackknife_indexes(data[0]) # I use the scikits.bootstrap function here.
    jstat = [statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
    jmean = np.mean(jstat,axis=0)

    # Acceleration value
    a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )
    if np.any(np.isnan(a)):
        nanind = np.nonzero(np.isnan(a))
        warnings.warn("Some acceleration values were undefined. \
            This is almost certainly because all values \
            for the statistic were equal. Affected \
            confidence intervals will have zero width and \
            may be inaccurate (indexes: {}). \
            Other warnings are likely related.".format(nanind))
    zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals = norm.cdf(z0 + zs/(1-a*zs))
    nvals = np.round((reps-1)*avals)
    nvals = np.nan_to_num(nvals).astype('int')
    
    return nvals
       
def bootstrap(data, 
              statfunction = None,
              smoothboot = True,
              alpha = 0.05, 
              reps = 5000):
    
    # Taken from scikits.bootstrap code
    # Initialise statfunction
    if statfunction == None:
        statfunction = np.mean
    
    # Compute two-sided alphas.
    alphas = np.array([alpha/2, 1-alpha/2])
    
    # Turns data into array, then tuple.
    data = np.array(data)
    tdata = (data,)

    # The value of the statistic function applied just to the actual data.
    ostat = statfunction(*tdata)
    
    ## Convenience function invoked to get array of desired bootstraps see above!
    # statarray = getstatarray(tdata, statfunction, reps, sort = True)
    statarray = sb.bootstrap(data, func = statfunction, n_boot = reps, smooth = smoothboot)
    statarray.sort()

    # Get Percentile indices
    pct_low_high = np.round((reps-1)*alphas)
    pct_low_high = np.nan_to_num(pct_low_high).astype('int')

    # Get Bias-Corrected Accelerated indices convenience function invoked.
    bca_low_high = bca(tdata, alphas, statarray, statfunction, ostat, reps)
    
    # Warnings for unstable or extreme indices.
    for ind in [pct_low_high, bca_low_high]:
        if np.any(ind==0) or np.any(ind==reps-1):
            warnings.warn("Some values used extremal samples results are probably unstable.")
        elif np.any(ind<10) or np.any(ind>=reps-10):
            warnings.warn("Some values used top 10 low/high samples results may be unstable.")
        
    result = OrderedDict()
    result['summary'] = ostat
    result['statistic'] = str(statfunction)
    result['bootstrap_reps'] = reps
    result['pct_ci_low'] = statarray[pct_low_high[0]]
    result['pct_ci_high'] = statarray[pct_low_high[1]]
    result['bca_ci_low'] = statarray[bca_low_high[0]]
    result['bca_ci_high'] = statarray[bca_low_high[1]]
    result['stat_array'] = np.array(statarray)
    result['pct_low_high_indices'] = pct_low_high
    result['bca_low_high_indices'] = bca_low_high
    return result

def bootstrap_contrast(data = None,
                       idx = None,
                       x = None,
                       y = None,
                       statfunction = None,
                       smoothboot = True,
                       alpha = 0.05, 
                       reps = 2000):
    
    # Taken from scikits.bootstrap code
    # Initialise statfunction
    if statfunction == None:
        statfunction = np.mean
    # check if idx was parsed
    if idx == None:
        idx = [0,1]
        
    # Compute two-sided alphas.
    alphas = np.array([alpha/2, 1-alpha/2])
    
    levels = data[x].unique()

    # Two types of dictionaries
    levels_to_idx = dict( zip(list(levels), range(0,len(levels))) ) # levels are the keys.
    idx_to_levels = dict( zip(range(0,len(levels)), list(levels)) ) # level indexes are the keys.
                                                                    # Not sure if I need this latter dict.
    
    # The loop approach below allows us to mix and match level and indices
    # when declaring the idx above.
    arraylist = list() # list to temporarily store the rawdata arrays.
    for i in idx:
        if i in levels_to_idx: # means the supplied id is an actual level
            arraylist.append( np.array(data.ix[data[x] == levels[levels_to_idx[i]]][y]) ) # when I get levels
        elif i in idx_to_levels: # means the supplied id is the level index (does this make sense?)
            arraylist.append( np.array(data.ix[data[x] == levels[i]][y]) ) # when I get level indexes
            
    # Pull out the arrays. 
    # The first array in `arraylist` is the reference array. 
    # Turn into tuple, so can iterate? Not sure.
    #ref_array = (arraylist[0],)
    #exp_array = (arraylist[1],)
    ref_array = arraylist[0]
    exp_array = arraylist[1]
    
    # Generate statarrays for both arrays
    #ref_statarray = getstatarray(ref_array, statfunction, reps, sort = False)
    ref_statarray = sb.bootstrap(ref_array, func = statfunction, n_boot = reps, smooth = smoothboot)
    #exp_statarray = getstatarray(exp_array, statfunction, reps, sort = False)
    exp_statarray = sb.bootstrap(exp_array, func = statfunction, n_boot = reps, smooth = smoothboot)
    
    diff_array = exp_statarray - ref_statarray
    diff_array_t = (diff_array,) # Note tuple form.
    diff_array.sort()

    # The difference as one would calculate it.
    ostat = statfunction(exp_array) - statfunction(ref_array)
    
    # Get Percentile indices
    pct_low_high = np.round((reps-1)*alphas)
    pct_low_high = np.nan_to_num(pct_low_high).astype('int')

    # Get Bias-Corrected Accelerated indices convenience function invoked.
    bca_low_high = bca(diff_array_t, alphas, diff_array, statfunction, ostat, reps)
    
    # Warnings for unstable or extreme indices.
    for ind in [pct_low_high, bca_low_high]:
        if np.any(ind==0) or np.any(ind==reps-1):
            warnings.warn("Some values used extremal samples results are probably unstable.")
        elif np.any(ind<10) or np.any(ind>=reps-10):
            warnings.warn("Some values used top 10 low/high samples results may be unstable.")
            
    # two-tailed t-test to see if the means of both arrays are different.
    ttestresult = ttest_ind(arraylist[0], arraylist[1])
    
    # Mann-Whitney test to see if the mean of the diff_array is not zero.
    mannwhitneyresult = mannwhitneyu(arraylist[0], arraylist[1])
    
    result = OrderedDict()
    result['summary'] = ostat
    result['statistic'] = str(statfunction)
    result['bootstrap_reps'] = reps
    result['pct_ci_low'] = diff_array[pct_low_high[0]]
    result['pct_ci_high'] = diff_array[pct_low_high[1]]
    result['bca_ci_low'] = diff_array[bca_low_high[0]]
    result['bca_ci_high'] = diff_array[bca_low_high[1]]
    result['diffarray'] = np.array(diff_array)
    result['pct_low_high_indices'] = pct_low_high
    result['bca_low_high_indices'] = bca_low_high
    result['statistic_ref'] = statfunction(ref_array)
    result['statistic_exp'] = statfunction(exp_array)
    result['ref_input'] = arraylist[0]
    result['test_input'] = arraylist[1]
    result['pvalue_ttest'] = ttestresult[1]
    result['pvalue_mannWhitney'] = mannwhitneyresult[1] * 2 # two-sided test result.
    return result

def plotbootstrap(coll, bslist, ax, violinWidth, 
                  violinOffset, marker = 'o', color = 'k', 
                  markerAlpha = 0.75,
                  markersize = 12,
                  CiAlpha = 0.75,
                  offset = True,
                  linewidth = 2, 
                  rightspace = 0.2,
                 **kwargs):
    # subfunction to plot the bootstrapped distribution along with BCa intervals.
    
    autoxmin = ax.get_xlim()[0]
    x, _ = np.array(coll.get_offsets()).T
    xmax = x.max()

    if offset:
        violinbasex = xmax + violinOffset
    else:
        violinbasex = 1
        
    array = list(bslist.items())[7][1]
    
    v = ax.violinplot(array, [violinbasex], 
                      widths = violinWidth * 2, 
                      showextrema = False, showmeans = False)
    
    for b in v['bodies']:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color('k')
    
    # Plot the summary measure.
    ax.plot(violinbasex, bslist['summary'],
             marker = marker,
             markerfacecolor = color, 
             markersize = 12,
             alpha = markerAlpha
            )

    # Plot the CI.
    ax.plot([violinbasex, violinbasex],
             [bslist['bca_ci_low'], bslist['bca_ci_high']],
             color = color, 
             alpha = CiAlpha,
             linestyle = 'solid'
            )
            
    ##  summary line
    #ax.plot([violinbasex, violinbasex + violinWidth], 
    #        [bslist['summary'], bslist['summary']], color, linewidth=2)

    ##  mean CI
    #ax.plot([violinbasex, violinbasex + violinWidth/3], 
    #        [bslist['bca_ci_low'], bslist['bca_ci_low']], color, linewidth)
    #ax.plot([violinbasex, violinbasex + violinWidth/3], 
    #        [bslist['bca_ci_high'], bslist['bca_ci_high']], color, linewidth)
    #ax.plot([violinbasex, violinbasex], 
    #        [bslist['bca_ci_low'], bslist['bca_ci_high']], color, linewidth)
    
    ax.set_xlim(autoxmin, (violinbasex + violinWidth + rightspace))
    
    if array.min() < 0 < array.min():
        ax.set_ylim(array.min(), array.max())
    elif 0 <= array.min(): 
        ax.set_ylim(0, array.max() * 1.1)
    elif 0 >= array.max():
        ax.set_ylim(array.min() * 1.1, 0)
        
def plotbootstrap_hubspoke(bslist, ax, violinWidth, violinOffset, 
                           marker = 'o', color = 'k', 
                           markerAlpha = 0.75,
                           markersize = 12,
                           CiAlpha = 0.75,
                           linewidth = 2,
                          **kwargs):
    
    # subfunction to plot the bootstrapped distribution along with BCa intervals for hub-spoke plots.
    ylims = list()
    
    for i in range(0, len(bslist)):
        bsi = bslist[i]
        array = list(bsi.items())[7][1] # Pull out the bootstrapped array.
        ylims.append(array)
        
        # Then plot as violinplot.
        v = ax.violinplot(array, [i+1], 
                          widths = violinWidth * 2, 
                          showextrema = False, showmeans = False)
        
        for b in v['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color('k')
            # Plot the summary measure.
            ax.plot(i+1, bsi['summary'],
                     marker = marker,
                     markerfacecolor = color, 
                     markersize = 12,
                     alpha = markerAlpha
                    )

            # Plot the CI.
            ax.plot([i+1, i+1],
                     [bsi['bca_ci_low'], bsi['bca_ci_high']],
                     color = color, 
                     alpha = CiAlpha,
                     linestyle = 'solid'
                    )

            ##  summary line
            #ax.plot([i+1, i+1 + violinWidth], 
            #        [bsi['summary'], bsi['summary']], color, linewidth=2)
            ##  mean CI
            #ax.plot([i+1, i+1 + violinWidth/3], 
            #        [bsi['bca_ci_low'], bsi['bca_ci_low']], color, linewidth)
            #ax.plot([i+1, i+1 + violinWidth/3], 
            #        [bsi['bca_ci_high'], bsi['bca_ci_high']], color, linewidth)
            #ax.plot([i+1, i+1], 
            #        [bsi['bca_ci_low'], bsi['bca_ci_high']], color, linewidth)
            
    ylims = np.array(ylims).flatten()
    if ylims.min() < 0 and ylims.max() < 0: # All effect sizes are less than 0.
        ax.set_ylim(1.1 * ylims.min(), 0)
    elif ylims.min() > 0:                   # All effect sizes are more than 0.
        ax.set_ylim(-0.25, 1.1 * ylims.max())
    elif ylims.min() < 0 < ylims.max():     # One or more effect sizes straddle 0.
        ax.set_ylim(1.1 * ylims.min(), 1.1 * ylims.max())

def swarmsummary(data, x, y, idx = None, statfunction = None, 
                 violinOffset = 0.1, violinWidth = 0.2, 
                 figsize = (7,7), legend = True,
                 smoothboot = True,
                 **kwargs):
    df = data # so we don't re-order the rawdata!
    # initialise statfunction
    if statfunction == None:
        statfunction = np.mean
        
    # calculate bootstrap list.
    bslist = OrderedDict()

    if idx is None:
        levs = df[x].unique()   # DO NOT USE the numpy.unique() method.
                                # It will not preserve the order of appearance of the levels.
    else:
        levs = idx

    for i in range (0, len(levs)):
        temp_df = df.loc[df[x] == levs[i]]
        bslist[levs[i]] = bootstrap(temp_df[y], statfunction = statfunction, smoothboot = smoothboot)
    
    bsplotlist = list(bslist.items())
    
    # Initialise figure
    #sb.set_style('ticks')
    fig, ax = plt.subplots(figsize = figsize)
    sw = sb.swarmplot(data = df, x = x, y = y, order = levs, **kwargs)
    y_lims = list()
    
    for i in range(0, len(bslist)):
        plotbootstrap(sw.collections[i], 
                      bslist = bsplotlist[i][1], 
                      ax = ax, 
                      violinWidth = violinWidth, 
                      violinOffset = violinOffset,
                      color = 'k', 
                      linewidth = 2)
        
        # Get the y-offsets, save into a list.
        _, y = np.array(sw.collections[i].get_offsets()).T 
        y_lims.append(y)
    
    # Concatenate the list of y-offsets
    y_lims = np.concatenate(y_lims)
    ax.set_ylim(0.9 * y_lims.min(), 1.1 * y_lims.max())
    
    if legend is True:
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 1))
    elif legend is False:
        ax.legend().set_visible(False)
        
    sb.despine(ax = ax, trim = True)
    
    return fig, pd.DataFrame.from_dict(bslist)
    

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    # Taken from 
    # http://stackoverflow.com/questions/7630778/matplotlib-align-origin-of-right-axis-with-specific-left-axis-value
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim() 
    ax2.set_ylim(miny+dy, maxy+dy)
    
def contrastplot(data, x, y, idx = None, statfunction = None, reps = 5000,
                 violinOffset = 0.375, violinWidth = 0.2, lineWidth = 2, pal = None,
                 summaryLineWidth = 0.25, figsize = None, 
                 heightRatio = (1, 1), alpha = 0.75,
                 showMeans = True, showMedians = False, 
                 showCI = False,  legend = True, 
                 meansColour = 'black', mediansColour = 'black', 
                 meansSummaryLineStyle = 'dashed', mediansSummaryLineStyle = 'dotted',
                 floatContrast = True, smoothboot = True, floatSwarmSpacer = 0.2,
                 effectSizeYLabel = "Effect Size", swarmShareY = True, contrastShareY = True,
                 **kwargs):
    
    # Set clean style
    sb.set_style('ticks')

    # initialise statfunction
    if statfunction == None:
        statfunction = np.mean
        
    # Set palette based on total number of categories in data['x'] or data['hue_column']
    if 'hue' in kwargs:
        u = kwargs['hue']
    else:
        u = x
        
    if pal is None:
        plotPal = dict( zip( data[u].unique(), sb.color_palette(n_colors = len(data[u].unique())) ) 
                      )
    else:
        plotPal = dict( zip( data[u].unique(), sb.color_palette(palette = pal, 
                                                                n_colors = len(data[u].unique())) ) 
                      )
        
    # Get and set levels of data[x]    
    if idx is None:
        # No idx is given, so all groups are compared to the first one in the DataFrame column.
        levs_tuple = (tuple(data[x].unique()), )
        widthratio = [1]
        if len(data[x].unique()) > 2:
            floatContrast = False
    else:
        # check if multi-plot or not
        if all(isinstance(element, str) for element in idx):
            # if idx is supplied but not a multiplot (ie single list or tuple) 
            levs_tuple = (idx, )
            widthratio = [1]
            if len(idx) > 2:
                floatContrast = False
        elif all(isinstance(element, tuple) for element in idx):
            # if idx is supplied, and it is a list/tuple of tuples or lists, we have a multiplot!
            levs_tuple = idx
            if (any(len(element) > 2 for element in levs_tuple) and floatContrast == True):
                # if any of the tuples in idx has more than 2 groups, we turn set floatContrast as False.
                floatContrast = False
            
            # Make sure the widthratio of the seperate multiplot corresponds to how 
            # many groups there are in each one.
            widthratio = []
            for i in levs_tuple:
                widthratio.append(len(i))
    
    # Create list to collect all the contrast DataFrames generated.
    contrastList = list()
    contrastListNames = list()
    
    if figsize is None:
        if len(levs_tuple) > 2:
            figsize = (12,(12/np.sqrt(2)))
        elif len(levs_tuple) < 2:
            figsize = (8,(8/np.sqrt(2)))
        
    # Initialise figure, taking into account desired figsize.
    fig = plt.figure(figsize = figsize)
    
    # Initialise GridSpec based on `levs_tuple` shape.
    gsMain = gridspec.GridSpec( 1, np.shape(levs_tuple)[0], # 1 row; columns based on number of tuples in tuple.
                               width_ratios = widthratio ) 
    
    for gsIdx, levs in enumerate(levs_tuple):
        # Create temp copy of the data for plotting!
        plotdat = data.copy()
        
        # Make sure the 'x' column is a 'category' type.
        plotdat[x] = plotdat[x].astype("category")
        plotdat = plotdat[plotdat[x].isin(levs)]
        plotdat[x].cat.set_categories(levs, ordered = True, inplace = True)
        
        # then order according to `levs`!
        plotdat.sort_values(by = [x])
        
        # Calculate means
        means = plotdat.groupby([x], sort = True).mean()[y]
        # Calculate medians
        medians = plotdat.groupby([x], sort = True).median()[y]

        if len(levs) == 2:            
            # Calculate bootstrap contrast. 
            tempbs = bootstrap_contrast(data = data, 
                                        x = x, 
                                        y = y,
                                        idx = levs, 
                                        statfunction = statfunction, 
                                        smoothboot = smoothboot,
                                        reps = reps)
            
            contrastListNames.append( str(levs[1]) + " v.s " + str(levs[0]) )
            contrastList.append(tempbs)

            if floatContrast is True:
                ax_left = fig.add_subplot(gsMain[gsIdx], frame_on = False) 
                # Use fig.add_subplot instead of plt.Subplot
                
                # Plot the raw data as a swarmplot.
                if showCI is True:
                    sb.barplot(data = plotdat, x = x, y = y, 
                               ax = ax_left, 
                               alpha = 0, 
                               ci = 95,
                               **kwargs)

                sw = sb.swarmplot(data = plotdat, x = x, y = y, 
                                  order = levs, ax = ax_left, 
                                  alpha = alpha, palette = plotPal,
                                  **kwargs)
                
                maxXBefore = max(sw.collections[0].get_offsets().T[0])
                minXAfter = min(sw.collections[1].get_offsets().T[0])

                xposAfter = maxXBefore + floatSwarmSpacer
                xAfterShift = minXAfter - xposAfter
                offsetSwarmX(sw.collections[1], -xAfterShift)
                
                ## Set the ticks locations for ax_left.
                axLeftLab = ax_left.get_xaxis().get_ticklabels
                ax_left.get_xaxis().set_ticks((0, xposAfter))
                ## Set the tick labels!
                ax_left.set_xticklabels([ax_left.get_xaxis().get_ticklabels()[0].get_text(),
                                         ax_left.get_xaxis().get_ticklabels()[1].get_text()],
                                       rotation = 45)
                ## Remove left axes x-axis title.
                ax_left.set_xlabel("")

                # Get the y-range of the data for y-axis limits
                y_lims = list()
                for i in range(0, 2):
                    # Get the y-offsets, save into a list.
                    _, ytemp = np.array(sw.collections[i].get_offsets()).T 
                    y_lims.append(ytemp)

                # Concatenate the list of y-offsets
                y_lims = np.concatenate(y_lims)
                #sb.despine(ax = ax_left, trim = True)

                # Set up floating axis on right.
                ax_right = ax_left.twinx()
                #ax_right = fig.add_subplot(gsMain[gsIdx], 
                #                           sharex = ax_left, 
                #                           frameon = False) 
                ## This allows ax_left to be seen beneath ax_right.

                # Then plot the bootstrap
                # We should only be looking at sw.collections[1],
                # as contrast plots will only be floating in this condition.
                plotbootstrap(sw.collections[1],
                              bslist = tempbs, 
                              ax = ax_right,
                              violinWidth = violinWidth, 
                              violinOffset = violinOffset,
                              color = 'k', 
                              linewidth = 2)
                ## If the effect size is positive, shift the right axis up.
                if float(tempbs['summary']) > 0:
                    rightmin = ax_left.get_ylim()[0] - float(tempbs['summary'])
                    rightmax = ax_left.get_ylim()[1] - float(tempbs['summary'])
                ## If the effect size is negative, shift the right axis down.
                elif float(tempbs['summary']) < 0:
                    rightmin = ax_left.get_ylim()[0] + float(tempbs['summary'])
                    rightmax = ax_left.get_ylim()[1] + float(tempbs['summary'])

                ## Lastly, align the mean of group 1 with the y = 0 of ax_right.
                ax_right.set_ylim(rightmin, rightmax)
                align_yaxis(ax_left, float(tempbs['statistic_ref']), ax_right, 0)

                # Set reference lines
                ## First get leftmost limit of left reference group
                xtemp, _ = np.array(sw.collections[0].get_offsets()).T
                leftxlim = xtemp.min()
                ## Then get leftmost limit of right test group
                xtemp, _ = np.array(sw.collections[1].get_offsets()).T
                rightxlim = xtemp.min()

                ## zero line
                ax_right.hlines(0,                   # y-coordinates
                                leftxlim, 3.5,       # x-coordinates, start and end.
                                linestyle = 'dotted')

                ## effect size line
                ax_right.hlines(tempbs['summary'], 
                                rightxlim, 3.5,        # x-coordinates, start and end.
                                linestyle = 'dotted') 

                if legend is True:
                    ax_left.legend(loc='center left', bbox_to_anchor=(1.1, 1))
                elif legend is False:
                    ax_left.legend().set_visible(False)
                    
                if gsIdx > 0:
                    ax_right.set_ylabel('')
                    
                # Trim the floating axes y-axis to an appropriate range around the bootstrap.
                ## Get the step size of the left axes y-axis.
                leftAxesStep = ax_left.get_yticks()[1] - ax_left.get_yticks()[0]
                ## figure out the number of decimal places for `leftStep`.
                dp = -Decimal(format(leftAxesStep)).as_tuple().exponent
                floatFormat = '.' + str(dp) + 'f'
                strLeftAxesStep = format(leftAxesStep)
                floatYMin = float(format(min(tempbs['diffarray']), floatFormat)) - leftAxesStep/2
                floatYMax = float(format(max(tempbs['diffarray']), floatFormat)) + leftAxesStep/2
                if floatYMin > 0.:
                    floatYMin = 0.
                if floatYMax < 0.:
                    floatYMax = 0.
                    
                ax_right.yaxis.set_ticks( np.arange(floatYMin,
                                                    floatYMax,
                                                    leftAxesStep/2) )

                sb.despine(ax = ax_left, trim = True)
                sb.despine(ax = ax_right, top = True, right = False, 
                           left = True, bottom = True, 
                           trim = True)

            elif floatContrast is False:
                # Create subGridSpec with 2 rows and 1 column.
                gsSubGridSpec = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                                                 subplot_spec = gsMain[gsIdx])
                ax_top = plt.Subplot(fig, gsSubGridSpec[0, 0], frame_on = False)

                if showCI is True:
                    sb.barplot(data = plotdat, x = x, y = y, ax = ax_top, alpha = 0, ci = 95)

                # Plot the swarmplot on the top axes.
                sw = sb.swarmplot(data = plotdat, x = x, y = y, 
                                  order = levs, ax = ax_top, 
                                  alpha = alpha, palette = plotPal,
                                  **kwargs)

                # Then plot the summary lines.
                if showMeans is True:
                    for i, m in enumerate(means):
                        ax_top.plot((i - summaryLineWidth, i + summaryLineWidth),           # x-coordinates
                                    (m, m),                                                 # y-coordinates
                                    color = meansColour, linestyle = meansSummaryLineStyle)

                if showMedians is True:
                    for i, m in enumerate(medians):
                        ax_top.plot((i - summaryLineWidth, i + summaryLineWidth), 
                                    (m, m), 
                                    color = mediansColour, linestyle = mediansSummaryLineStyle)
                        
                if legend is True:
                    ax_top.legend(loc='center left', bbox_to_anchor=(1.1, 1))
                elif legend is False:
                    ax_top.legend().set_visible(False)
                    
                fig.add_subplot(ax_top)
                ax_top.set_xlabel('')
                #sb.despine(ax = ax_top, trim = True)
                
                # Initialise bottom axes
                ax_bottom = plt.Subplot(fig, gsSubGridSpec[1, 0], sharex = ax_top, frame_on = False)

                # Plot the CIs on the bottom axes.
                plotbootstrap(sw.collections[1],
                              bslist = tempbs,
                              ax = ax_bottom, 
                              violinWidth = violinWidth,
                              offset = False,
                              violinOffset = 0,
                              linewidth = 2)

                # Add zero reference line on bottom axes.
                ax_bottom.hlines(y = 0,
                                 xmin = ax_bottom.get_xlim()[0], 
                                 xmax = ax_bottom.get_xlim()[1],
                                 linestyle = 'dotted')

                # Set bottom axes ybounds
                ax_bottom.set_ybound( tempbs['diffarray'].min(), tempbs['diffarray'].max() )
                
                # Set xlims so everything is properly visible!
                swarm_xbounds = ax_top.get_xbound()
                ax_bottom.set_xbound(swarm_xbounds[0] - (summaryLineWidth * 1.1), 
                                     swarm_xbounds[1] + (summaryLineWidth * 1.))

                # Equalize the top and bottom sets of axes.
                
                # Label the bottom y-axis
                fig.add_subplot(ax_bottom)
                ax_bottom.set_ylabel(effectSizeYLabel)
                sb.despine(ax = ax_top, trim = True)
                sb.despine(ax = ax_bottom, left = False, bottom = False, trim = True)
                
                if gsIdx > 0:
                    ax_top.set_ylabel('')
                    ax_bottom.set_ylabel('')
                    
        elif len(levs) > 2:
            bscontrast = list()
            # Create subGridSpec with 2 rows and 1 column.
            gsSubGridSpec = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                                     subplot_spec = gsMain[gsIdx])
                        
            # Calculate the hub-and-spoke bootstrap contrast.

            for i in range (1, len(levs)): # Note that you start from one. No need to do auto-contrast!
                tempbs = bootstrap_contrast(data = data,
                                            x = x, 
                                            y = y, 
                                            idx = [levs[0], levs[i]],
                                            statfunction = statfunction, 
                                            smoothboot = smoothboot,
                                            reps = reps)
                bscontrast.append(tempbs)
                contrastList.append(tempbs)
                contrastListNames.append(levs[i] + ' vs. ' + levs[0])

            # Initialize the top swarmplot axes.
            ax_top = plt.Subplot(fig, gsSubGridSpec[0, 0], frame_on = False)
            
            if showCI is True:
                sb.barplot(data = plotdat, x = x, y = y, ax = ax_top, alpha = 0, ci = 95)

            sw = sb.swarmplot(data = plotdat, x = x, y = y, 
                              order = levs, ax = ax_top, 
                              alpha = alpha, palette = plotPal,
                              **kwargs)

            # Then plot the summary lines.
            if showMeans is True:
                for i, m in enumerate(means):
                    ax_top.plot((i - summaryLineWidth, i + summaryLineWidth),            # x-coordinates
                                (m, m),                                                  # y-coordinates
                                color = meansColour, linestyle = meansSummaryLineStyle)

            if showMedians is True:
                for i, m in enumerate(medians):
                    ax_top.plot((i - summaryLineWidth, i + summaryLineWidth), 
                                (m, m), 
                                color = mediansColour, linestyle = mediansSummaryLineStyle)

            if legend is True:
                ax_top.legend(loc='center left', bbox_to_anchor=(1.1, 1))
            elif legend is False:
                ax_top.legend().set_visible(False)
            
            fig.add_subplot(ax_top)
            ax_top.set_xlabel('')

            # Initialise the bottom swarmplot axes.
            ax_bottom = plt.Subplot(fig, gsSubGridSpec[1, 0], sharex = ax_top, frame_on = False)
            
            # Plot the CIs on the bottom axes.
            plotbootstrap_hubspoke(bslist = bscontrast,
                                   ax = ax_bottom, 
                                   violinWidth = violinWidth,
                                   violinOffset = violinOffset,
                                   linewidth = lineWidth)

            # Add zero reference line on bottom axes.
            ax_bottom.hlines(y = 0,
                             xmin = ax_bottom.get_xlim()[0], 
                             xmax = ax_bottom.get_xlim()[1],
                             linestyle = 'dotted')
            
            # Set xlims so everything is properly visible!
            swarm_xbounds = ax_top.get_xbound()
            ax_bottom.set_xbound(swarm_xbounds[0] - (summaryLineWidth * 1.1), 
                                 swarm_xbounds[1] + (summaryLineWidth * 1.))
            
            # Label the bottom y-axis
            fig.add_subplot(ax_bottom)
            ax_bottom.set_ylabel(effectSizeYLabel)
            sb.despine(ax = ax_top, trim = True)
            sb.despine(ax = ax_bottom, left = False, bottom = False, trim = True)
            
            if gsIdx > 0:
                ax_top.set_ylabel('')
                ax_bottom.set_ylabel('')
            
    # Turn contrastList into a pandas DataFrame
    contrastList = pd.DataFrame(contrastList).T
    contrastList.columns = contrastListNames
    
    # Normalize top/left axes to each other
    if (len(fig.get_axes()) > 2 and swarmShareY is True):
        normalizeSwarmY(fig)
    
    # Normalize bottom/right axes to each other.
    if (len(fig.get_axes()) > 2 and contrastShareY is True):
        normalizeContrastY(fig, floatContrast)
    
    if swarmShareY is False:
        for i in range(0, len(fig.get_axes()), 2):
            sb.despine(ax = fig.get_axes()[i], trim = True)
                       
    if contrastShareY is False:
        for i in range(1, len(fig.get_axes()), 2):
            if floatContrast is True:
                sb.despine(ax = fig.get_axes()[i], 
                           top = True, right = False, left = True, bottom = True, 
                           trim = True)
            else:
                sb.despine(ax = fig.get_axes()[i], trim = True)
    
    # Draw back the lines for the leftmost swarm and contrast y-axes.
    # So we get the axes we need to re-draw.
    if floatContrast is False:
        ix = (0,1)
    else:
        #ix = list(range(1, len(fig.get_axes()), 2))
        #ix.append(0)
        ix = [0]
    
    for i in ix:
        if (floatContrast is False or i == 0):
            x, _ = fig.get_axes()[i].get_xaxis().get_view_interval()
        else:
            _, x = fig.get_axes()[i].get_xaxis().get_view_interval()
            
        ymin = fig.get_axes()[i].get_yaxis().get_majorticklocs()[0]
        ymax = fig.get_axes()[i].get_yaxis().get_majorticklocs()[-1]
        fig.get_axes()[i].add_artist(Line2D((x, x), (ymin, ymax), color='black', linewidth=2))
        
    # Draw back the lines for all the x-axis.
    for i in range(0, len(fig.get_axes()) ):
        xmin = fig.get_axes()[i].get_xaxis().get_majorticklocs()[0]
        xmax = fig.get_axes()[i].get_xaxis().get_majorticklocs()[-1]
        y, _ = fig.get_axes()[i].get_yaxis().get_view_interval()
        
        fig.get_axes()[i].add_artist(Line2D((xmin, xmax), (y, y), color='black', linewidth=2))
        
    # Tight Layout!
    gsMain.tight_layout(fig)
    
    # And we're all done.
    return fig, contrastList

def pairedcontrast(data, x, y, idcol, hue = None,
    statfunction = None, xlevs = None,
    beforeAfterSpacer = 0.1, violinWidth = 0.2, 
    swarmDeltaOffset = 0.3, floatOffset = 0.3, 
    violinOffset = 0.2,
    floatViolinOffset = 0.1,
    legendLoc = 2, legendFontSize = 12, legendMarkerScale = 1,
    **kwargs):

    # Sanity checks below.
    ## If `xlevs` is not specified, just take the FIRST TWO levels alphabetically.
    if xlevs is None:
        xlevs = np.unique(data[x])[0:2]
    elif xlevs is not None:
        if len(xlevs) != 2:
            print("xlevs does not have length 2.")
            sys.exit(1)
        
    if statfunction is None:
        statfunction = np.mean

    ## Start plotting!!
    fig = plt.figure(figsize = (8,8))
    ax_left = plt.subplot(111)

    swarm_raw = sb.swarmplot(data = data, 
                             x = x, y = y, 
                             hue = hue,
                             order = xlevs,
                             axes = ax_left, 
                             **kwargs)
    # ax_left.set_ylim( (round(min(data[y])), 
    #                    round(max(data[y]))) ) # Set a tight y-limit.

    if hue is not None:
        swarm_raw.legend(loc = legendLoc, 
            fontsize = legendFontSize, 
            markerscale = legendMarkerScale)

    # Shifting the `after` raw swarmplot to appropriate xposition.
    maxXBefore = max(swarm_raw.collections[0].get_offsets().T[0])
    minXAfter = min(swarm_raw.collections[1].get_offsets().T[0])

    xposAfter = maxXBefore + beforeAfterSpacer
    xAfterShift = minXAfter - xposAfter
    offsetSwarmX(swarm_raw.collections[1], -xAfterShift)

    # pivot the original df!
    data_pivot = data.pivot(index = idcol, columns = x, values = y) 

    # create convenient signifiers for column names.
    befX = str(xlevs[0] + '_x')
    aftX = str(xlevs[1] + '_x')

    # pandas DataFrame of 'before' group
    x1 = pd.DataFrame({str(xlevs[0] + '_x') : pd.Series(swarm_raw.collections[0].get_offsets().T[0]),
                   xlevs[0] : pd.Series(swarm_raw.collections[0].get_offsets().T[1]),
                   'R' : pd.Series(swarm_raw.collections[0].get_facecolors().T[0]),
                   'G' : pd.Series(swarm_raw.collections[0].get_facecolors().T[1]),
                   'B' : pd.Series(swarm_raw.collections[0].get_facecolors().T[2]),
                  })
    # join the RGB columns into a tuple, then assign to a column.
    x1['hue'] = x1[['R', 'G', 'B']].apply(tuple, axis=1) 
    x1 = x1.sort_values(by = xlevs[0])
    x1.index = data_pivot.sort_values(by = xlevs[0]).index

    # pandas DataFrame of 'after' group
    x2 = pd.DataFrame( {aftX : pd.Series(swarm_raw.collections[1].get_offsets().T[0]),
        xlevs[1] : pd.Series(swarm_raw.collections[1].get_offsets().T[1])} )
    x2 = x2.sort_values(by = xlevs[1])
    x2.index = data_pivot.sort_values(by = xlevs[1]).index

    # Join x1 and x2, on both their indexes.
    plotPoints = x1.merge(x2, left_index = True,right_index = True, how='outer')

    # Add the hue column if hue argument was passed.
    if hue is not None:
        plotPoints[hue] = data.pivot(index = idcol, columns = x, values = hue)[xlevs[0]]

    # Plot the lines to join the 'before' points to their respective 'after' points.
    for i in plotPoints.index:
        ax_left.plot([ plotPoints.ix[i, befX],
            plotPoints.ix[i, aftX] ],
            [ plotPoints.ix[i, xlevs[0]], 
            plotPoints.ix[i, xlevs[1]] ],
            linestyle = 'solid',
            color = plotPoints.ix[i, 'hue'],
            alpha = 0.25
            )
        
    for i in (0,1):
        # Calculate the boostrapped mean and 95% CI for before and after,
        # for each of the swarmplot groups.
        points = pd.DataFrame( swarm_raw.collections[i].get_offsets() )
        # second column of `points` is the y-values, which is what we want.
        # run bootstrap on it.
        bootsRaw = bootstrap(points[1], statfunction = statfunction)
        summRaw = bootsRaw['summary']
        lowRaw = bootsRaw['bca_ci_low']
        highRaw = bootsRaw['bca_ci_high']
        
        # Get the x-position of the summary line/violin.
        if i == 0:
            xpos = min(points[0]) - violinOffset
            before_xpos = xpos
        if i == 1:
            xpos = max(points[0]) + violinOffset
            after_xpos = xpos
        
        # Plot the summary measure.
        plt.plot(xpos, summRaw,
                 marker = 'D',
                 markerfacecolor = 'k', 
                 markersize = 12,
                 alpha = 0.75
                )
        
        # Plot the CI.
        plt.plot([xpos, xpos],
                 [lowRaw, highRaw],
                 color = 'k', 
                 alpha = 0.75,
                 linestyle = 'solid'
                )
        
        # Plot the violin-plot.
        v = swarm_raw.violinplot(bootsRaw['stat_array'], [xpos], 
                                 widths = violinWidth * 2, 
                                 showextrema = False, 
                                 showmeans = False)
        
        if i == 0:
            # show left-half of violin
            halfviolin(v, right = False)
        if i == 1:
            # show right-half of violin
            halfviolin(v, right = True)

    # Generate floating axes on right
    ax_float = ax_left.twinx()
        
    # Calculate the summary difference and CI.
    plotPoints['delta_y'] = plotPoints[xlevs[1]] - plotPoints[xlevs[0]]
    plotPoints['delta_x'] = [0] * np.shape(plotPoints)[0]

    bootsDelta = bootstrap(plotPoints['delta_y'] , statfunction = statfunction)
    summDelta = bootsDelta['summary']
    lowDelta = bootsDelta['bca_ci_low']
    highDelta = bootsDelta['bca_ci_high']

    # Plot the delta swarmplot.
    deltaSwarm = sb.swarmplot(data = plotPoints,
        x = 'delta_x',
        y = 'delta_y',
        marker = '^',
        hue = hue,
        **kwargs)
    # Make sure they have the same x-limits and y-limits.
    ax_float.set_xlim(ax_left.get_xlim())
    ax_float.set_ylim(ax_left.get_ylim())

    # Shifting the delta swarmplot to appropriate xposition
    xposPlus = xposAfter + swarmDeltaOffset + (violinWidth * 2)         
    offsetSwarmX(deltaSwarm.collections[0], xposPlus)

    # set new xpos for delta violin.
    xposPlusViolin = deltaSwarmX = xposPlus + floatViolinOffset

    # Plot the summary measure.
    plt.plot(xposPlusViolin, summDelta,
             axes = ax_float,
             marker = 'o',
             markerfacecolor = 'k', 
             markersize = 12,
             alpha = 0.75
            )

    # Plot the CI.
    plt.plot([xposPlusViolin, xposPlusViolin],
             [lowDelta, highDelta],
             axes = ax_float,
             color = 'k', 
             alpha = 0.75,
             linestyle = 'solid'
            )

    # Plot the violin-plot.
    v = ax_float.violinplot(bootsDelta['stat_array'], [xposPlusViolin], 
                             widths = violinWidth * 2, 
                             showextrema = False, 
                             showmeans = False)
    halfviolin(v, right = True, color = 'k')

    # Set reference lines
    ## zero line
    ax_float.hlines(0,                                    # y-coordinate
                    before_xpos, xposPlusViolin + 0.25,   # x-coordinates, start and end.
                    linestyle = 'dotted')

    ## effect size line
    ax_float.hlines(summDelta, 
                    after_xpos, xposPlusViolin + 0.25,
                    linestyle = 'dotted') 

    # Set xlimit to appropriate limits..
    newxlim = (ax_left.get_xlim()[0], xposPlusViolin + 0.25)
    ax_left.set_xlim(newxlim)

    # Remove left axes x-axis title.
    ax_left.set_xlabel("")
    # Remove floating axes y-axis title.
    ax_float.set_ylabel("")
    # Turn off hue legend for floating axes.
    if hue is not None:
        ax_float.legend().set_visible(False)

    # Drawing in the x-axis for ax_left.
    ## Get lowest y-value for ax_left.
    y = ax_left.get_yaxis().get_view_interval()[0] 
    ## Set the ticks locations for ax_left.
    ax_left.get_xaxis().set_ticks((0, xposAfter))
    ## Set the tick labels!
    ax_left.set_xticklabels(xlevs, rotation = 45, horizontalalignment = 'right')

    # Align the left axes and the floating axes.
    align_yaxis(ax_left, statfunction(plotPoints[xlevs[0]]),
                   ax_float, 0)

    # Add label to ax_float.
    ax_float.text(x = deltaSwarmX - floatViolinOffset,
                  y = ax_float.get_yaxis().get_view_interval()[0],
                  horizontalalignment = 'left',
                  s = 'Difference',
                  fontsize = 15)

    # Trim the floating axes y-axis to an appropriate range around the bootstrap.
    ## Get the step size of the left axes y-axis.
    leftAxesStep = ax_left.get_yticks()[1] - ax_left.get_yticks()[0]
    ## figure out the number of decimal places for `leftStep`.
    dp = -Decimal(format(leftAxesStep)).as_tuple().exponent
    floatFormat = '.' + str(dp) + 'f'
    floatYMin = float(format(min(bootsDelta['stat_array']), 
                             floatFormat)) - leftAxesStep/2
    floatYMax = float(format(max(bootsDelta['stat_array']), 
                             floatFormat)) + leftAxesStep/2
    if floatYMin > 0.:
        floatYMin = 0.
    if floatYMax < 0.:
        floatYMax = 0.
        
    ax_float.yaxis.set_ticks( np.arange(floatYMin,
                                        floatYMax,
                                        leftAxesStep/2) )

    plt.tight_layout()

    # Despine all axes.
    sb.despine(ax = ax_left, trim = True)
    sb.despine(ax = ax_float, right = False, left = True, top = True, trim = True)

    ## And we're done.
    return fig, dictToDf(bootsDelta, x)

def normalizeSwarmY(fig):
    allYmax = list()
    allYmin = list()
    
    for i in range(0, len(fig.get_axes()), 2):
        # First, loop thru the axes and compile a list of their ybounds (aka y-limits).
        allYmin.append(fig.get_axes()[i].get_ybound()[0])
        allYmax.append(fig.get_axes()[i].get_ybound()[1])

    # Then loop thru the axes again to equalize them.
    for i in range(0, len(fig.get_axes()), 2):
        fig.get_axes()[i].set_ybound(np.min(allYmin), np.max(allYmax))
        
        if (i > 0 and len(fig.get_axes()) > 2) :
            # remove all but the leftmost swarm axes if it is a multiplot.
            fig.get_axes()[i].get_yaxis().set_visible(False)        

def normalizeContrastY(fig, floatContrast):
    allYmax = list()
    allYmin = list()
    
    # First, loop thru the axes and compile a list of their ybounds (aka y-limits).
    for i in range(1, len(fig.get_axes()), 2):
        allYmin.append(fig.get_axes()[i].get_ybound()[0])
        allYmax.append(fig.get_axes()[i].get_ybound()[1])
        
    # Then loop thru the axes again to equalize them.
    for i in range(1, len(fig.get_axes()), 2):
        fig.get_axes()[i].set_ybound(np.min(allYmin), np.max(allYmax))
        
        # If they are floating....
        if (floatContrast is True and i > 1):
            # Remove the superfluous x-axis for the floating contrast axes as well.
            fig.get_axes()[i].get_xaxis().set_visible(False)
                                     
        # ... but if they are not floating...
        if floatContrast is False:
            if (i > 1 and len(fig.get_axes()) > 2):
                # ... and remove all but the leftmost one.
                fig.get_axes()[i].get_yaxis().set_visible(False)

def contrastplot_match(figlist = [], top_ybounds = None, bottom_ybounds = None):
    ## This function is not used anymore.... but never delete any code, no?
    # Get the max top ybounds
    if top_ybounds is None:
        topY = list()
        topYscales = list()
        for f in figlist:
            topY.append(f.get_axes()[0].get_ybound())
            r = f.get_axes()[0].get_ybound()[1] - f.get_axes()[0].get_ybound()[0]
            topYscales.append(r)
            
        topY = np.array(topY).flatten()
        top_ybounds = (topY.min(), topY.max())
    
    if bottom_ybounds is None:
        bottomY = list()
        bottomYscales = list()
        for f in figlist:
            bottomY.append(f.get_axes()[1].get_ybound())
            r = f.get_axes()[0].get_ybound()[1] - f.get_axes()[0].get_ybound()[0]
            bottomYscales.append(r)
            
        bottomY = np.array(bottomY).flatten()
        bottom_ybounds = (bottomY.min(), bottomY.max())
    
    max_top_idx = np.argmax(topYscales)
    max_bottom_idx = np.argmax(bottomYscales)
    
    for f in figlist:
        f.get_axes()[0].set_ybound( top_ybounds )
        f.get_axes()[1].set_ybound( bottom_ybounds )
        
        f.get_axes()[0].set_yticks( figlist[max_top_idx].get_axes()[0].get_yticks() )
        f.get_axes()[1].set_yticks( figlist[max_bottom_idx].get_axes()[1].get_yticks() )
        
        sb.despine(ax = f.get_axes()[0], trim = True)
        sb.despine(ax = f.get_axes()[1], trim = True)
    
    return figlist

def halfviolin(v, right = True, color = 'k'):
    for b in v['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            if right is False:
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            if right is True:
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color(color)
            
def offsetSwarmX(c, xpos):
    newx = c.get_offsets().T[0] + xpos
    newxy = np.array([list(newx), list(c.get_offsets().T[1])]).T
    c.set_offsets(newxy)

def resetSwarmX(c, newxpos):
    lengthx = len(c.get_offsets().T[0])
    newx = [newxpos] * lengthx
    newxy = np.array([list(newx), list(c.get_offsets().T[1])]).T
    c.set_offsets(newxy)

def dictToDf(df, name):
	# convenience function to convert orderedDict object to pandas DataFrame.
	# args: df is an orderedDict, name is a string.
	l = list()
	l.append(df)
	l_df = pd.DataFrame(l).T
	l_df.columns = [name]
	return l_df
    