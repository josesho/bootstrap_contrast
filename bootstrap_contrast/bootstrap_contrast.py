from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
# perform version check.
# The apionly has been deprecated after version 8.
sns_version=sns.__version__.split('.')
sns_version_major=int(sns_version[0])
sns_version_minor=int(sns_version[1])

if sns_version_major==0 or sns_version_minor<8:
    import seaborn.apionly as sns

# import scikits.bootstrap as skb

import pandas as pd
import numpy as np
import scipy as sp

from .plot_tools import halfviolin, align_yaxis, rotate_ticks
from .bootstrap_tools import bootstrap, jackknife_indexes, bca

def contrastplot(data, idx,
             x=None, y=None, 
             color_col=None,
             
             float_contrast=True,
             paired=False, 
             show_pairs=True,
             
             swarm_ylim=None,
             contrast_ylim=None,
             
             fig_size=None,
             
             font_scale=1.4,
             
             stat_func=np.mean,
             ci=95,n_boot=5000,
             
             show_group_count=True,

             custom_palette=None,
             
             swarmplot_kwargs=None,
             violinplot_kwargs=None,
             reflines_kwargs=None,
             legend_kwargs=None,
             palette_kwargs=None,
             aesthetic_kwargs=None,

            ):
    
    '''Takes a pandas DataFrame and produces a contrast plot:
    either a Cummings hub-and-spoke plot or a Gardner-Altman contrast plot.
    Paired and unpaired options available.

    Keyword arguments:
        data: pandas DataFrame

        idx: tuple
            List of column names (if 'x' is not supplied) or of category names (if 'x' is supplied).
            This can be expressed as a tuple of tuples, with each individual tuple producing its own
            contrast plot.

        x, y: strings, default None

        color_col: list, default None
            List of colors (either named matplotlib colors or RGB tuples) to be used to color the
            different categories.

        float_contrast: boolean, default True
            Whether or not to display the halfviolin bootstrapped difference distribution 
            alongside the raw data.

        paired: boolean, default False
            Whether or not the data is paired. To elaborate.

        show_pairs: boolean, default True
            If the data is paired, whether or not to show the raw data as a swarmplot, or as 
            paired plot, with a line joining each pair of observations.

        swarm_ylim: tuple, default None
            The desired y-limits of the raw data swarmplot as a (lower, higher) tuple.

        contrast_ylim: tuple, default None
            The desired y-limits of the constrast plot as a (lower, higher) tuple.

        fig_size: tuple, default None
            The desired dimensions of the figure as a (length, width) tuple.

        font_scale: float, default 1.4
            The font size will be scaled by this number.

        stat_func: callable, default np.mean
            The function used to compute the summary.

        ci: integer, default 95
            The size of the confidence interval desired (in percentage).

        n_boot: integer, default 5000
            Number of bootstrap iterations to perform during calculation of confidence intervals.

        show_group_count: boolean, default True
            Whether or not the group count (e.g. 'N=10') will be appended to the xtick labels. 

        custom_palette: dict, default None

        swarmplot_kwargs: dict, default None

        violinplot_kwargs: dict, default None

        reflines_kwargs: dict, default None

        legend_kwargs: dict, default None

        palette_kwargs: dict, default None

        aesthetic_kwargs: dict, default None

     Returns:
        A matplotlib Figure.
        Arrangement of the axes. Obtaining access to them via figure.axes[i]

        A pandas DataFrame.
        Description of column headings.

    '''
    
    ### MAKE COPY OF DATA.
    data_in=data.copy()
    data_in.reset_index(inplace=True)

    ### IDENTIFY PLOT TYPE.
    if all([isinstance(i, str) for i in idx]):
        # plottype='hubspoke'
        allgrps=np.unique([t for t in idx]) # flatten out idx.
        idx=(idx,)
        ncols=1
        widthratio=[1]
        if len(idx)>2:
            paired=False
            float_contrast=False
    elif all([isinstance(i, tuple) for i in idx]):
        # plottype='multiplot'
        allgrps=np.unique([tt for t in idx for tt in t])
        ncols=len(idx)
        widthratio=[]
        for i in idx:
            if len(i)>1:
                widthratio.append(len(i))
                if len(i)>2:
                    paired=False
                    float_contrast=False
            else:
                raise ValueError('One of the tuples in `idx` only has one category. Please check.')

    ### SANITY CHECKS
    # check color_col is a column name.
    if (color_col is not None) and (color_col not in data_in.columns):
        raise IndexError('The specified `color_col` '+color_col+' is not a column in `data`. Please check.')

    if x is not None and y is not None:
        # Assume we have a long dataset.
        datatype='long'
        # check both x and y are column names in data.
        if x not in data_in.columns:
            raise IndexError(x+' is not a column in `data`. Please check.')
        if y not in data_in.columns:
            raise IndexError(y+'is not a column in `data`. Please check.')
        # check y is numeric.
        if not np.issubdtype(data_in[y].dtype, np.number):
            raise ValueError(y+'is a column in `data`, but it is not numeric. Please check.')
    elif x is None and y is None:
        # Assume we have a wide dataset.
        datatype='wide'
        # extract only the columns we need.
        ## first check we have all columns in the dataset.
        for g in allgrps:
            if g not in data_in.columns:
                raise IndexError(g+' is not a column in `data`. Please check.')
        ## Melt it so it is easier to use.
        x='group'
        y='value'
        if color_col is None:
            idv=['index']
        else:
            idv=['index',color_col]
        data_in=pd.melt(data_in.reset_index(),
                        id_vars=idv,
                        value_vars=allgrps,
                        value_name=x, 
                        var_name=y)
        idv.append(x)
        idv.append(y)
        data_in.columns=[idv]
    elif x is None and y is not None:
        raise ValueError('You have only specified `y`. Please also specify `x`')
    elif y is None and x is not None:
        raise ValueError('You have only specified `x`. Please also specify `y`')
    
    # CALCULATE CI.
    if ci<0 or ci>100:
        raise ValueError('`ci` should be between 0 and 100.')
    alpha_level=(100.-int(ci))/100.

    # CALCULATE RAW SWARM YLIMS.
    if swarm_ylim is None:
        pad=np.abs(data_in[y].diff().min())/2 # To ensure points at the limits are clearly seen.
        swarm_ylim=( data_in[y].min()-pad, 
                    data_in[y].max()+pad )
    
    ### PARSE kwargs.
    if (swarmplot_kwargs is None) or ('size' not in swarmplot_kwargs.keys()):
        if swarmplot_kwargs==None:
            swarmplot_kwargs={}
        swarmplot_kwargs['size']=5
    
    if (violinplot_kwargs is None) or ('widths' not in violinplot_kwargs.keys()):
        if violinplot_kwargs is None:
            violinplot_kwargs={'widths':0.5,
                               'vert':True,
                               'showextrema':False,
                               'showmedians':False}
        violinplot_kwargs['widths']=0.5

    if reflines_kwargs is None:
        reflines_kwargs={'linestyle':'solid',
                         'linewidth':0.75,
                         'color':'k'}
        
    if (legend_kwargs is None) or ('markerscale' not in legend_kwargs.keys()):
        if legend_kwargs is None:
            legend_kwargs={}
        legend_kwargs['markerscale']=1.25
        
    if (palette_kwargs is None) or ('n_colors' not in palette_kwargs.keys()):
        if palette_kwargs is None:
            palette_kwargs={}
        palette_kwargs['n_colors']=len(allgrps)

    if aesthetic_kwargs is None:
        aesthetic_kwargs={'context':'poster','style':'ticks','font_scale':font_scale}
        
    # if paired is False, set show_pairs as False.
    if paired is False:
        show_pairs=False

    ### INITIALISE FIGURE.
    # Set clean style.
    sns.set(**aesthetic_kwargs)
    # Set appropriate horizontal spacing between subplots, based on whether the contrast is floating.
    if float_contrast:
        ws=0.75
    else:
        ws=0.
    # Set figure size.
    if fig_size is None:
        if len(idx)>2:
            fig_size=(11, (11/np.sqrt(2)) )
        else:
            fig_size=(7, (7/np.sqrt(2)) )
    # Create subplots.
    fig,axx=plt.subplots(ncols=ncols,figsize=fig_size,gridspec_kw={'wspace':ws,
                                                                  'width_ratios':widthratio})
    # If the contrast axes are NOT floating, create lists to store raw ylims
    # and raw tick intervals, so that I can normalize their ylims later.
    if float_contrast is False:
        contrast_ax_ylim_low=list()
        contrast_ax_ylim_high=list()
        contrast_ax_ylim_tickintervals=list()
        
    ### CREATE COLOR PALETTE TO NORMALIZE PALETTE ACROSS AXES.
    if color_col is None:
        col_grp=x
    else:
        col_grp=color_col
    colGrps=data_in[col_grp].unique()
    if custom_palette is None:
        plotPal=dict( zip( colGrps, sns.color_palette(**palette_kwargs) ) )
    else:
        # ensure that number of colors is equal or greater than number of groups.
        if len(colGrps)>len(custom_palette):
            raise ValueError('The number of colors supplied to `custom_palette` is less than the total number of desired color groups.')
        plotPal=dict( zip(colGrps, custom_palette) )
    
    ### LIST TO STORE BOOTSTRAPPED RESULTS.
    bootlist=list()
    
    ### FOR EACH TUPLE IN IDX, CREATE PLOT.
    for j, current_tuple in enumerate(idx):
        plotdat=data_in[data_in[x].isin(current_tuple)]
        plotdat.loc[:,x]=plotdat[x].astype("category")
        plotdat[x].cat.set_categories(
            current_tuple,
            ordered=True,
            inplace=True)
        plotdat.sort_values(by=[x])
        summaries=plotdat.groupby(x)[y].apply(stat_func)
        # Compute Ns per group.
        counts=plotdat.groupby(x)[y].count()

        if ncols==1:
            ax_raw=axx
        else:
            ax_raw=axx[j]
        
        if float_contrast:
            ax_contrast=ax_raw.twinx()
        else:
            divider=make_axes_locatable(ax_raw)
            ax_contrast=divider.append_axes("bottom", size="100%", pad=0.5, sharex=ax_raw)
        
        ### PLOT RAW DATA.
        ax_raw.set_ylim(swarm_ylim)
        if paired:
            # first, sanity checks. Do we have 2 elements (no more, no less) here?
            if len(current_tuple)!=2:
                raise ValueError('Paired plotting is True, but '+str(current_tuple)+'does not have 2 elements.')
            # Are the groups equal in length??
            before=plotdat[ plotdat[x]==current_tuple[0] ][y].dropna().tolist()
            after=plotdat[ plotdat[x]==current_tuple[1] ][y].dropna().tolist()
            if len(before)!=len(after):
                raise ValueError('The sizes of '+current_tuple[0]+' and '+current_tuple[1]+' do not match.')
                
        if paired and show_pairs:
            if color_col is not None:
                colors=plotdat[plotdat[x]==current_tuple[0]][color_col]
            else:
                plotPal['__default_black__']=(0., 0., 0.) # black
                colors=np.repeat('__default_black__',len(before))
            linedf=pd.DataFrame(
                    {str(current_tuple[0]):before,
                    str(current_tuple[1]):after,
                    'colors':colors}
                    )
            # Set xticklabels for plot.
            # to get color, need to loop thru each line and plot individually.
#             for c in linedf.colors.unique():
#                 temp_linedf=linedf[linedf.colors==c]
            for ii in linedf.index:
                ax_raw.plot( [0,1],  # x1, x2
                            [ linedf.loc[ii,current_tuple[0]],
                             linedf.loc[ii,current_tuple[1]] ], # y1, y2
                            linestyle='solid',
                            color=plotPal[ linedf.loc[ii,'colors'] ],
                            linewidth=0.5,
#                             label=linedf.loc[ii,'colors']
#                             linewidth=pairedDeltaLineWidth,
#                             color=plotPal[current_tuple[0]],
#                             alpha=pairedDeltaLineAlpha,
                           )
            ax_raw.set_ylabel(y)
            ax_raw.set_xticks([0,1])
            ax_raw.set_xticklabels( [current_tuple[0],current_tuple[1]] )
            
        elif (paired is True and show_pairs is False) or (paired is False):
            sns.swarmplot(data=plotdat, 
                          x=x, y=y, 
                          order=current_tuple, 
                          ax=ax_raw,
                          hue=color_col,
                          palette=plotPal,
                          **swarmplot_kwargs)
        ax_raw.set_xlabel('')
        
        # Set new tick labels. The tick labels belong to the SWARM axes
        # for both floating and non-floating plots. This is because `sharex` was invoked.
        newticklabs=list()
        for xticklab in ax_raw.xaxis.get_ticklabels():
            t=xticklab.get_text()
            N=str(counts.ix[t])
            if show_group_count:
                newticklabs.append(t+' n='+N)
            else:
                newticklabs.append(t)
            ax_raw.set_xticklabels(newticklabs,rotation=45,horizontalalignment='right')
        
        # Despine appropriately.    
        if float_contrast:
            sns.despine(ax=ax_raw,trim=True)
        else:
            ax_raw.xaxis.set_visible(False)
            not_first_ax=(j!=0)
            sns.despine(ax=ax_raw,bottom=True,left=not_first_ax,trim=True)
            if not_first_ax:
                ax_raw.yaxis.set_visible(False)
        
        ### PLOT CONTRAST DATA.
        # Calculate bootstrapped stats.
        # Plot the CIs on the bottom axes.
        for ix, grp in enumerate( current_tuple[1::] ) :
            # add spacer to halfviolin if float_contast is true.
            if float_contrast is True:
                spacer=0.75
#             elif (paired is True and show_pairs is True):
#                 spacer=0.5
            else:
                spacer=0
            pos=ix+spacer
            # Calculate bootstrapped stats.
            boots=bootstrap(np.array(plotdat[plotdat[x]==current_tuple[0]][y].dropna()),
                np.array(plotdat[plotdat[x]==grp][y].dropna()),
                paired=paired,
                alpha_level=alpha_level,
                statfunction=stat_func,
                reps=n_boot)
            res=boots.results
            res['reference_group']=current_tuple[0]
            res['experimental_group']=grp
            res['pvalue_1samp_ttest']=boots.pvalue_1samp_ttest
            res['pvalue_2samp_ind_ttest']=boots.pvalue_2samp_ind_ttest
            res['pvalue_2samp_paired_ttest']=boots.pvalue_2samp_paired_ttest
            res['pvalue_wilcoxon']=boots.pvalue_wilcoxon
            res['pvalue_mannWhitney']=boots.pvalue_mannWhitney
            bootlist.append(res)
            # Plot the halfviolin and mean+CIs on contrast axes.
            v=ax_contrast.violinplot( boots.stat_array,
                                     positions=[pos+1],
                                     **violinplot_kwargs
                                    )
            halfviolin(v)
            ax_contrast.plot([pos+1], boots.summary, marker='o', color='k', markersize=10)
            ax_contrast.plot([pos+1,pos+1], [boots.bca_ci_low, boots.bca_ci_high], 'k-', linewidth=2)
            if float_contrast is False:
                contrast_ax_ylim_low.append( ax_contrast.get_ylim()[0] )
                contrast_ax_ylim_high.append( ax_contrast.get_ylim()[1] )
                ticklocs=ax_contrast.yaxis.get_majorticklocs()
                contrast_ax_ylim_tickintervals.append( ticklocs[1]-ticklocs[0] )
            
        ### NORMALISE Y LIMS AND DESPINE FLOATING CONTRAST AXES.
        if float_contrast:
            ## Align 0 of ax_contrast to reference group mean of ax_raw.
            ylimlow,ylimhigh=ax_contrast.get_xlim()
            ax_contrast.set_xlim(ylimlow, ylimhigh+spacer)
            # If the effect size is positive, shift the contrast axis up.
            if boots.summary>0:
                rightmin=ax_raw.get_ylim()[0]-boots.summary
                rightmax=ax_raw.get_ylim()[1]-boots.summary
            # If the effect size is negative, shift the contrast axis down.
            elif boots.summary<0:
                rightmin=ax_raw.get_ylim()[0]+boots.summary
                rightmax=ax_raw.get_ylim()[1]+boots.summary
            ax_contrast.set_ylim(rightmin, rightmax)
            align_yaxis(ax_raw, np.mean(plotdat[plotdat[x]==grp][y].dropna()), 
                           ax_contrast, boots.summary)
            # Draw zero line.
            xlimlow, xlimhigh=ax_contrast.get_xlim()
            ax_contrast.hlines(0,   # y-coordinates
                               0, xlimhigh,  # x-coordinates, start and end.
                               **reflines_kwargs)
            
            # Draw effect size line.
            ax_contrast.hlines(boots.summary, 
                               1, xlimhigh,  # x-coordinates, start and end.
                               **reflines_kwargs)
            
            ## Shrink or stretch axis to encompass 0 and min/max contrast.
            # Get the lower and upper limits.
            lower=boots.stat_array.min()
            upper=boots.stat_array.max()
            # Make sure we have zero in the limits.
            if lower>0:
                lower=0.
            if upper<0:
                upper=0.
            # Get the tick interval from the left y-axis.
            leftticks=ax_contrast.get_yticks()
            tickstep=leftticks[1] -leftticks[0]
            # First re-draw of axis with new tick interval
            ax_contrast.yaxis.set_major_locator(tk.MultipleLocator(base=tickstep))
            newticks1=ax_contrast.get_yticks()
            # Obtain major ticks that comfortably encompass lower and upper.
            newticks2=list()
            for a,b in enumerate(newticks1):
                if (b >= lower and b <= upper):
                    # if the tick lies within upper and lower, take it.
                    newticks2.append(b)
            # if the boots.summary falls outside of the newticks2 set, add a tick in the right direction.
            if np.max(newticks2)<boots.summary:
                ind=np.where(newticks1 == np.max(newticks2))[0][0] # find out the max tick index in newticks1.
                newticks2.append( newticks1[ind+1] )
            elif boots.summary<np.min(newticks2):
                ind=np.where(newticks1 == np.min(newticks2))[0][0] # find out the min tick index in newticks1.
                newticks2.append( newticks1[ind-1] )
            newticks2=np.array(newticks2)
            newticks2.sort()
            # Re-draw axis to shrink it to desired limits.
            ax_contrast.yaxis.set_major_locator(tk.FixedLocator(locs=newticks2))
            ## Despine the axes. 
            sns.despine(ax=ax_contrast, trim=True, 
                        left=True, bottom=True, # remove the left and bottom spines...
                        right=False) # ...but not the right spine.

        ### SET Y AXIS LABELS .   
        if float_contrast:
            if j>0:
                ax_raw.set_ylabel('')
        else:
            if j>0:
                ax_raw.set_ylabel('')
                ax_contrast.set_ylabel('')
            else:
                if paired:
                    ax_contrast.set_ylabel('paired contrast '+y)
                else:
                    ax_contrast.set_ylabel('contrast '+y)
                
        ### ONLY SHOW COLOR LEGEND FOR RIGHTMOST AXES.
        if color_col is not None:
            if j+1==ncols:
                if (paired is True) and (show_pairs is True):
                    handles_list=[]
                    for key, value in plotPal.items():
                        l=mlines.Line2D([], [], color=value, 
                                        label=key)
                        handles_list.append(l)
                    ax_raw.legend(handles=handles_list,
                                  loc='upper left', bbox_to_anchor=(1., 0.99),**legend_kwargs)
                else:
                    ax_raw.legend(loc='upper left', bbox_to_anchor=(1., 0.99),**legend_kwargs)
            else:
                if show_pairs is False:
                    ax_raw.legend().set_visible(False)
        ### ROTATE X-TICKS OF ax_contrast
        rotate_ticks(ax_contrast,angle=45,alignment='right')
        
    ### NORMALIZE Y LIMS AND DESPINE NON-FLOATING CONTRAST AXES.
    if float_contrast is False:
        ## Sort and convert to numpy arrays.
        contrast_ax_ylim_low=np.sort(contrast_ax_ylim_low)
        contrast_ax_ylim_high=np.sort(contrast_ax_ylim_high)
        contrast_ax_ylim_tickintervals=np.sort(contrast_ax_ylim_tickintervals)
        ## Compute normalized ylim, or set normalized ylim to desired ylim.
        if contrast_ylim is None:
            normYlim=(contrast_ax_ylim_low[0], contrast_ax_ylim_high[-1])
        else:
            normYlim=contrast_ylim
        ## Loop thru the contrast axes again to re-draw all the y-axes.
        for i in range(ncols, ncols*2, 1): 
            # The last half of the axes in `fig` are the contrast axes.
            axx=fig.get_axes()[i]
            # Set the axes to the max ylim
            axx.set_ylim(normYlim[0], normYlim[1])
            # Draw zero reference line if zero is in the ylim range.
            if normYlim[0]<0. and 0.<normYlim[1]:
                axx.axhline(y=0,lw=0.5,color='k')
            # Hide the y-axis except for the leftmost contrast axes.
            if i>ncols:
                axx.get_yaxis().set_visible(False)
                sns.despine(ax = axx, left=True, trim=True)
            else:
                # Despine.
                sns.despine(ax = axx, trim = True)

    ### PREPARE OUTPUT
    # Turn `bootlist` into a pandas DataFrame
    bootlist_df=pd.DataFrame(bootlist)
    # Order the columns properly.
    bootlist_df=bootlist_df[['reference_group', 'experimental_group','stat_summary',
    'bca_ci_low', 'bca_ci_high', 'ci', 
    'is_difference', 'is_paired', 
    'pvalue_1samp_ttest', 'pvalue_2samp_ind_ttest', 'pvalue_2samp_paired_ttest',
    'pvalue_mannWhitney', 'pvalue_wilcoxon',]]
    # Reset seaborn aesthetic parameters.
    sns.set()
    # Return the figure and the results DataFrame.
    return fig, bootlist_df
