from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, FixedLocator, AutoLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def rotateTicks(axes, angle, alignment='right'):
    for tick in axes.get_xticklabels():
        tick.set_rotation(angle)
        tick.set_horizontalalignment(alignment)

def normalizeSwarmY(fig, floatcontrast):
    allYmax = list()
    allYmin = list()
    
    for i in range(0, len(fig.get_axes()), 2):
        # First, loop thru the axes and compile a list of their ybounds.
        allYmin.append(fig.get_axes()[i].get_ybound()[0])
        allYmax.append(fig.get_axes()[i].get_ybound()[1])

    # Then loop thru the axes again to equalize them.
    for i in range(0, len(fig.get_axes()), 2):
        fig.get_axes()[i].set_ylim(np.min(allYmin), np.max(allYmax))
        fig.get_axes()[i].get_yaxis().set_view_interval(np.min(allYmin), np.max(allYmax))

        YAxisStep = fig.get_axes()[i].get_yticks()[1] - fig.get_axes()[i].get_yticks()[0]
        # Setup the major tick locators.
        majorLocator = MultipleLocator(YAxisStep)
        majorFormatter = FormatStrFormatter('%.1f')
        fig.get_axes()[i].yaxis.set_major_locator(majorLocator)
        fig.get_axes()[i].yaxis.set_major_formatter(majorFormatter)

        if (floatcontrast is False):
            sb.despine(ax = fig.get_axes()[i], top = True, right = True, 
                       left = False, bottom = True, 
                       trim = True)
        else:
            sb.despine(ax = fig.get_axes()[i], top = True, right = True, 
                   left = False, bottom = False, 
                   trim = True)

        # Draw back the lines for the relevant y-axes.
        x, _ = fig.get_axes()[i].get_xaxis().get_view_interval()
        ymin = fig.get_axes()[i].get_yaxis().get_majorticklocs()[0]
        ymax = fig.get_axes()[i].get_yaxis().get_majorticklocs()[-1]
        fig.get_axes()[i].add_artist(Line2D((x, x), (ymin, ymax), color='black', linewidth=1.5))
            

def normalizeContrastY(fig, con, contrast_ylim, show_all_yaxes):
    allYmax = list()
    allYmin = list()
    tickintervals = list()
    # loop thru the axes and compile a list of their y-axis tick intervals.
    # Then get the max tick interval.
    for i in range(1, len(fig.get_axes()), 2):
        ticklocs = fig.get_axes()[i].yaxis.get_majorticklocs()
        tickintervals.append(ticklocs[1] - ticklocs[0])
    maxTickInterval = np.max(tickintervals)

    if contrast_ylim is None:
        # If no ylim is specified,
        # loop thru the axes and compile a list of their ybounds (aka y-limits)
        for j,i in enumerate(range(1, len(fig.get_axes()), 2)):
            allYmin.append(fig.get_axes()[i].get_ybound()[0])
            allYmax.append(fig.get_axes()[i].get_ybound()[1])
            maxYbound = np.array([np.min(allYmin), np.max(allYmax)])
    else:
        maxYbound = contrast_ylim

    # Loop thru the contrast axes again to re-draw all the y-axes.
    for i in range(1, len(fig.get_axes()), 2):
        ## Set the axes to the max ybounds, or the specified contrast_ylim.

        fig.get_axes()[i].get_yaxis().set_view_interval(maxYbound[0], maxYbound[1])

        ## Setup the tick locators.
        majorLocator = MultipleLocator(maxTickInterval)
        fig.get_axes()[i].yaxis.set_major_locator(majorLocator)

        ## Reset the view interval to the limits of the major ticks.
        majorticklocs_y = fig.get_axes()[i].yaxis.get_majorticklocs()
        fig.get_axes()[i].get_yaxis().set_view_interval(majorticklocs_y[0], majorticklocs_y[-1])

        sb.despine(ax = fig.get_axes()[i], top = True, right = True, 
            left = False, bottom = True, 
            trim = True)

        ## Draw back the lines for the relevant x-axes.
        xmin = fig.get_axes()[i].get_xaxis().get_majorticklocs()[0]
        xmax = fig.get_axes()[i].get_xaxis().get_majorticklocs()[-1]
        y, _ = fig.get_axes()[i].get_yaxis().get_view_interval()
        fig.get_axes()[i].add_artist(Line2D((xmin, xmax), (y, y), color='black', linewidth=1.5))  

        ## Draw back the lines for the relevant y-axes.
        x, _ = fig.get_axes()[i].get_xaxis().get_view_interval()
        if show_all_yaxes is False:
            ## Draw the leftmost contrast y-axis in first...
            fig.get_axes()[1].add_artist(Line2D((x, x), 
                (majorticklocs_y[0], majorticklocs_y[-1]), color='black', linewidth=1.5))

            ## ... then hide the non left-most contrast y-axes.
            if i > 1:
                fig.get_axes()[i].get_yaxis().set_visible(False)

        else:
            ## If you want to see all contrast y-axes, draw in their lines.
            fig.get_axes()[i].add_artist(Line2D((x, x), 
                (majorticklocs_y[0], majorticklocs_y[-1]), color='black', linewidth=1.5))


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

def halfviolin(v, half = 'right', color = 'k'):
    for b in v['bodies']:
            mVertical = np.mean(b.get_paths()[0].vertices[:, 0])
            mHorizontal = np.mean(b.get_paths()[0].vertices[:, 1])
            if half is 'left':
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, mVertical)
            if half is 'right':
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], mVertical, np.inf)
            if half is 'bottom':
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], -np.inf, mHorizontal)
            if half is 'top':
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], mHorizontal, np.inf)
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

def getSwarmSpan(swarmplot, groupnum):
    return swarmplot.collections[groupnum].get_offsets().T[0].ptp(axis = 0)

def drawback_y(ax):
    # Draw back the lines for the relevant y-axes.
    x, _ = ax.get_xaxis().get_view_interval()
    ymin = ax.get_yaxis().get_majorticklocs()[0]
    ymax = ax.get_yaxis().get_majorticklocs()[-1]
    ax.add_artist(Line2D((x, x), (ymin, ymax), color='black', linewidth=1))

def drawback_x(ax):
    # Draw back the lines for the relevant x-axes.
    y, _ = ax.get_yaxis().get_view_interval()
    xmin = ax.get_xaxis().get_majorticklocs()[0]
    xmax = ax.get_xaxis().get_majorticklocs()[-1]
    ax.add_artist(Line2D((xmin, xmax), (y, y), color='black', linewidth=1))