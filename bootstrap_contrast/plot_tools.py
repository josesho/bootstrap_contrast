import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .misc_tools import merge_two_dicts


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

def rotate_ticks(axes, angle=45, alignment='right'):
    for tick in axes.get_xticklabels():
        tick.set_rotation(angle)
        tick.set_horizontalalignment(alignment)

def plot_means(data,x,y,ax=None,xwidth=0.5,zorder=1,linestyle_kw=None):
    """Takes a pandas DataFrame and plots the `y` means of each group in `x` as horizontal lines.

    Keyword arguments:
        data: pandas DataFrame.
            This DataFrame should be in 'wide' format.

        x,y: string.
            x and y columns to be plotted.

        xwidth: float, default 0.5
            The horizontal spread of the line. The default is 0.5, which means
            the mean line will stretch 0.5 (in data coordinates) on both sides
            of the xtick.

        zorder: int, default 1
            This is the plot order of the means on the axes.
            See http://matplotlib.org/examples/pylab_examples/zorder_demo.html

        linestyle_kw: dict, default None
            Dictionary with kwargs passed to the `meanprops` argument of `plt.boxplot`.
    """

    # Set default linestyle parameters.
    default_linestyle_kw=dict(
            linewidth=1.5,
            color='k',
            linestyle='-')
    # If user has specified kwargs for linestyle, merge with default params.
    if linestyle_kw is None:
        meanlinestyle_kw=default_linestyle_kw
    else:
        meanlinestyle_kw=merge_two_dicts(default_linestyle_kw,linestyle_kw)

    # Set axes for plotting.
    if ax is None:
        ax=plt.gca()

    # Use sns.boxplot to create the mean lines.
    sns.boxplot(data=data,
                x=x,y=y,
                ax=ax,
                showmeans=True,
                meanline=True,
                showbox=False,
                showcaps=False,
                showfliers=False,
                whis=0,
                width=xwidth,
                zorder=int(zorder),
                meanprops=meanlinestyle_kw,
                medianprops=dict(linewidth=0)
               )

def plot_std(data, x, y, offset=0, zorder=5, ax=None):
    if ax is None:
        ax = plt.gca()

    num_groups = len(data[x].unique())
    ax.errorbar(x=np.array(range(0, num_groups)) + offset,
                y=data.groupby(x)[y].mean().tolist(),
                yerr=data.groupby(x)[y].std().tolist(),
                lw=1.2,
                zorder=zorder,
                color='k',
                fmt='none')
