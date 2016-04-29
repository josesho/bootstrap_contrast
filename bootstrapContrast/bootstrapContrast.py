#import scikits.bootstrap as sciboots
from collections import OrderedDict
from scipy.stats import ttest_ind
from numpy.random import randint
import matplotlib.pyplot as plt
from scipy.stats import norm
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
    
def bootstrap_indexes(data, n_samples=10000):
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
    # convenience function for use within `bootstrap` and `bootstrap_contrast`.
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
              reps = 10000):
    
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
                       reps = 10000):
    
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
            
     # two-tailed t-test to see if the mean of the diff_array is not zero.
    ttestresult = ttest_ind(arraylist[0], arraylist[1])
    
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
    result['statistic_ref'] = statfunction(ref_array[0])
    result['statistic_exp'] = statfunction(exp_array[0])
    result['ref_input'] = arraylist[0]
    result['test_input'] = arraylist[1]
    result['pvalue_ttest'] = ttestresult[1]
    return result

def plotbootstrap(coll, bslist, ax, violinWidth, 
                  violinOffset, color = 'k', 
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
    
    #  summary line
    ax.plot([violinbasex, violinbasex + violinWidth], 
            [bslist['summary'], bslist['summary']], color, linewidth=2)

    #  mean CI
    ax.plot([violinbasex, violinbasex + violinWidth/3], 
            [bslist['bca_ci_low'], bslist['bca_ci_low']], color, linewidth)
    ax.plot([violinbasex, violinbasex + violinWidth/3], 
            [bslist['bca_ci_high'], bslist['bca_ci_high']], color, linewidth)
    ax.plot([violinbasex, violinbasex], 
            [bslist['bca_ci_low'], bslist['bca_ci_high']], color, linewidth)
    
    ax.set_xlim(autoxmin, (violinbasex + violinWidth + rightspace))
    if array.min() < 0 < array.min():
        ax.set_ylim(array.min(), array.max())
    elif 0 <= array.min(): 
        ax.set_ylim(0, array.max() * 1.1)
    elif 0 >= array.max():
        ax.set_ylim(array.min() * 1.1, 0)
        
def plotbootstrap_hubspoke(bslist, ax, violinWidth, violinOffset, color = 'k', linewidth = 2):
    
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
            
            #  summary line
            ax.plot([i+1, i+1 + violinWidth], 
                    [bsi['summary'], bsi['summary']], color, linewidth=2)
            #  mean CI
            ax.plot([i+1, i+1 + violinWidth/3], 
                    [bsi['bca_ci_low'], bsi['bca_ci_low']], color, linewidth)
            ax.plot([i+1, i+1 + violinWidth/3], 
                    [bsi['bca_ci_high'], bsi['bca_ci_high']], color, linewidth)
            ax.plot([i+1, i+1], 
                    [bsi['bca_ci_low'], bsi['bca_ci_high']], color, linewidth)
            
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
    sb.set_style('ticks')
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
    
def contrastplot(data, x, y, idx = None, statfunction = None, reps = 10000,
                 violinOffset = 0.1, violinWidth = 0.2, summaryLineWidth = 0.25,
                 figsize = (7,7), legend = True,
                 showMeans = True, showMedians = False,
                 meansColour = 'black', mediansColour = 'black',
                 meansSummaryLineStyle = 'dashed', mediansSummaryLineStyle = 'dotted',
                 floatContrast = True, smoothboot = True, alpha = 0.75,
                 **kwargs):
    
    # initialise statfunction
    if statfunction == None:
        statfunction = np.mean

    # Get and set levels of data[x]    
    if idx is None:
        levs = data[x].unique()
    else:
        levs = idx
    # Create temp data for plotting, where the levels are sorted according to idx.
    plotdat = data.copy()
    plotdat[x] = plotdat[x].astype("category")             
    plotdat[x].cat.set_categories(levs,
                                  ordered = True, 
                                  inplace = False)  # Sets the categories as idx,
    
    plotdat.sort([x])                           # then order according to idx!
    
    # Calculate means
    means = plotdat.groupby([x], sort = False).mean()[y]
    # Calculate medians
    medians = plotdat.groupby([x], sort = False).median()[y]
    
    # Calculate the contrast for each pairwise comparison
    if len(levs) > 2:
        # Calculate the hub-and-spoke bootstrap contrast.
        bscontrast = list()
        colnames = list()
        for i in range (1, len(levs)): # Note that you start from one. No need to do auto-contrast!
            bscontrast.append(bootstrap_contrast(data = plotdat,
                                                 x = x, 
                                                 y = y, 
                                                 idx = [levs[0],levs[i]], 
                                                 statfunction = statfunction, 
                                                 smoothboot = smoothboot,
                                                 reps = reps))
            colnames.append(levs[i])
        
        # Initialize the figure.
        fig, (ax_top, ax_bottom) = plt.subplots(2, sharex = True, figsize = figsize)
        sb.set_style('ticks')
       
        # Plot the swarmplot on the top axes.
        sw = sb.swarmplot(data = plotdat, x = x, y = y, 
                          order = levs, ax = ax_top, 
                          alpha = alpha,
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
        
        # Plot the CIs on the bottom axes.
        plotbootstrap_hubspoke(bslist = bscontrast,
                               ax = ax_bottom, 
                               violinWidth = violinWidth,
                               violinOffset = violinOffset,
                               linewidth = 2)

        # Add zero reference line on bottom axes.
        ax_bottom.hlines(y = 0,
                         xmin = ax_bottom.get_xlim()[0], 
                         xmax = ax_bottom.get_xlim()[1],
                         linestyle = 'dotted')
                
        # Label the bottom y-axis
        ax_bottom.set_ylabel('Effect Size')
        sb.despine(ax = ax_top, trim = True)
        sb.despine(ax = ax_bottom, trim = True)
        
        bscontrast = pd.DataFrame(bscontrast).T
        bscontrast.columns = colnames
        
    elif len(levs) == 2:
        # Calculate bootstrap contrast. 
        bscontrast = bootstrap_contrast(data = plotdat, 
                                        x = x, 
                                        y = y, 
                                        idx = levs, 
                                        statfunction = statfunction, 
                                        smoothboot = smoothboot,
                                        reps = reps)

        if floatContrast:
            fig = plt.figure(figsize = figsize)
            sb.set_style('ticks')
            ax_left = fig.add_subplot(111)

            # Plot the raw data as a swarmplot.
            sw = sb.swarmplot(data = plotdat, x = x, y = y, 
                              order = levs, ax = ax_left, 
                              alpha = alpha,
                              **kwargs)

            # Get the y-range of the data for y-axis limits
            y_lims = list()
            for i in range(0, 2):
                # Get the y-offsets, save into a list.
                _, y = np.array(sw.collections[i].get_offsets()).T 
                y_lims.append(y)

            # Concatenate the list of y-offsets
            y_lims = np.concatenate(y_lims)
            # Set the left axis y-limits
            ax_left.set_ylim(0.9 * y_lims.min(), 1.1 * y_lims.max())
            sb.despine(ax = ax_left, trim = True)
            
            # Set up floating axis on right.
            ax_right = fig.add_axes(ax_left.get_position(), 
                                    sharex = ax_left, 
                                    #xlim = ax_left.get_xlim(), 
                                    frameon = False) # This allows ax_left to be seen beneath ax_right.

            # Then plot the bootstrap
            # We should only be looking at sw.collections[1],
            # as contrast plots will only be floating in this condition.
            plotbootstrap(sw.collections[1],
                          bslist = bscontrast, 
                          ax = ax_right, 
                          violinWidth = violinWidth, 
                          violinOffset = violinOffset,
                          color = 'k', 
                          linewidth = 2)
            sb.despine(ax = ax_right, top = True, right = False, left = True, trim = True)

            ## If the effect size is positive, shift the right axis up.
            if float(bscontrast['summary']) > 0:
                rightmin = ax_left.get_ylim()[0] - float(bscontrast['summary'])
                rightmax = ax_left.get_ylim()[1] - float(bscontrast['summary'])
            ## If the effect size is negative, shift the right axis down.
            elif float(bscontrast['summary']) < 0:
                rightmin = ax_left.get_ylim()[0] + float(bscontrast['summary'])
                rightmax = ax_left.get_ylim()[1] + float(bscontrast['summary'])

            ## Lastly, align the mean of group 1 with the y = 0 of ax_right.
            ax_right.set_ylim(rightmin, rightmax)
            align_yaxis(ax_left, float(bscontrast['statistic_ref']), ax_right, 0)
            
            # Set reference lines
            ## First get leftmost limit of left reference group
            x, _ = np.array(sw.collections[0].get_offsets()).T
            leftxlim = x.min()
            ## Then get leftmost limit of right test group
            x, _ = np.array(sw.collections[1].get_offsets()).T
            rightxlim = x.min()
            
            ## zero line
            ax_right.hlines(0,                   # y-coordinates
                            leftxlim, 3.5,       # x-coordinates, start and end.
                            linestyle = 'dotted')

            ## effect size line
            ax_right.hlines(bscontrast['summary'], 
                            rightxlim, 3.5,        # x-coordinates, start and end.
                            linestyle = 'dotted') 
            
            if legend is True:
                ax_left.legend(loc='center left', bbox_to_anchor=(1.1, 1))
            elif legend is False:
                ax_left.legend().set_visible(False)

        else:
            # Initialize the figure.
            fig, (ax_top, ax_bottom) = plt.subplots(2, sharex = True, figsize = figsize)
            sb.set_style('ticks')

            # Plot the swarmplot on the top axes.
            sw = sb.swarmplot(data = plotdat, x = x, y = y, 
                              order = levs, ax = ax_top, 
                              alpha = alpha,
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

            # Plot the CIs on the bottom axes.
            plotbootstrap(sw.collections[1],
                          bslist = bscontrast,
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

            # Label the bottom y-axis
            ax_bottom.set_ylabel('Effect Size')
            sb.despine(ax = ax_top, trim = True)
            sb.despine(ax = ax_bottom, trim = True)
            
        # turn `bscontrast` into list for easy conversion to pandas DataFrame.
        dftitle = str(levs[1]) + " v.s " + str(levs[0])
        bscontrast = pd.DataFrame.from_dict( {dftitle : bscontrast} )
            
    return fig, bscontrast
