
## Load Libraries


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import bootstrap_contrast as bs

import pandas as pd
import numpy as np
import scipy as sp
```

    /Users/josesho/anaconda3/envs/py3.6/lib/python3.6/site-packages/seaborn/apionly.py:6: UserWarning: As seaborn no longer sets a default style on import, the seaborn.apionly module is deprecated. It will be removed in a future version.
      warnings.warn(msg, UserWarning)


## Create dummy dataset

Here, we create a dummy dataset to illustrate how `bootstrap-contrast` functions.
In this dataset, each column corresponds to a group of observations, and each row is simply an index number referring to an observation. (This is known as a 'wide' dataset.)


```python
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
```

Note that we have 6 groups of observations, with an additional non-numerical column indicating gender.

## The `bootstrap` class

In `bootstrap-contrast` v0.3, I introduce a new class called `bootstrap`. Essentially, it will compute the summary statistic and its associated confidence interval using bootstrapping. It can do this for a single group of observations, or for two groups of observations (both paired and unpaired).

Below, I obtain the bootstrapped contrast for 'Control' and 'Group1' in `df`.


```python
contr=bs.bootstrap(df['Control'],df['Group1'])
```

As mentioned above, `contr` is a `bootstrap` object. Calling it directly will not produce anything.


```python
contr
```




    <bootstrap_contrast.bootstrap_tools.bootstrap at 0x1140a6e10>



It has several callable attributes. Of interest is its `results` attribute, which returns a dictionary summarising the results of the contrast computation.


```python
contr.results
```




    {'bca_ci_high': 0.21259206591324031,
     'bca_ci_low': -0.61533232157660067,
     'ci': 95.0,
     'is_difference': True,
     'is_paired': False,
     'stat_summary': -0.1808044652703821}



`is_paired` indicates the two arrays are paired (or repeated) observations. This is indicated by the `paired` flag.


```python
contr_paired=bs.bootstrap(df['Control'],df['Group1'],
                         paired=True)
contr_paired.results
```




    {'bca_ci_high': 0.22966945997396326,
     'bca_ci_low': -0.593827920249279,
     'ci': 95.0,
     'is_difference': True,
     'is_paired': True,
     'stat_summary': -0.18080446527038205}



`is_difference` basically indicates if one or two arrays were passed to the `bootstrap` function. Obseve what happens if we just give one array.


```python
just_control_=bs.bootstrap(df['Control'])
just_control_.results
```




    {'bca_ci_high': 0.46606211929057134,
     'bca_ci_low': -0.13429004281762066,
     'ci': 95.0,
     'is_difference': False,
     'is_paired': False,
     'stat_summary': 0.17175621510073041}



Here, the confidence interval is with respect to the mean of the group `Control`.


There are several other statistics the `bootstrap` object contains. Please do have a look at its documentation. Below, I print the p-values for `contr_paired` as an example.



```python
contr_paired.pvalue_2samp_paired_ttest
```




    0.39310007728828344




```python
contr_paired.pvalue_wilcoxon
```




    0.35369319267722144



## Producing Plots

Version 0.3 of `bootstrap-contrast` has an optimised version of the `contrastplot` command.

### Floating contrast plots—Two-group unpaired

Below we produce three aligned Gardner-Altman floating contrast plots.

The `contrastplot` command will return 2 objects: a matplotlib `Figure` and a pandas `DataFrame`.
In the Jupyter Notebook, with `%matplotlib inline`, the figure should automatically appear.

`bs.bootstrap` will automatically drop any NaNs in the data. Note how the Ns (appended to the group names in the xtick labels) indicate the number of datapoints being plotted, and used to calculate the contrasts.

The pandas `DataFrame` returned by `bs.bootstrap` contains the pairwise comparisons made in the course of generating the plot, with confidence intervals (95% by default) and relevant p-values.


```python
f,b=bs.contrastplot(df,
                    idx=('Control','Group1'),
                    color_col='Gender',
                    fig_size=(4,6) # The length and width of the image, in inches.
                   )
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_mannWhitney</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.59053</td>
      <td>0.225173</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.395987</td>
      <td>0.363178</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_21_1.png)


### Floating contrast plots—Two-group paired


```python
f,b=bs.contrastplot(df,
                    idx=('Control','Group2'),
                    color_col='Gender',
                    paired=True,
                    fig_size=(4,6))
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_paired_ttest</th>
      <th>pvalue_wilcoxon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group2</td>
      <td>-0.532006</td>
      <td>-1.010791</td>
      <td>-0.030387</td>
      <td>95.0</td>
      <td>True</td>
      <td>True</td>
      <td>0.04253</td>
      <td>0.038456</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_23_1.png)


If you want to plot the raw swarmplot instead of the paired lines, use the `show_pairs` flag to set this. The contrasts computed will still be paired, as indicated by the DataFrame produced.


```python
f,b=bs.contrastplot(df,
                    idx=('Control','Group2'),
                    color_col='Gender',
                    paired=True,
                    show_pairs=False,
                    fig_size=(4,6))
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_paired_ttest</th>
      <th>pvalue_wilcoxon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group2</td>
      <td>-0.532006</td>
      <td>-0.995463</td>
      <td>-0.029925</td>
      <td>95.0</td>
      <td>True</td>
      <td>True</td>
      <td>0.04253</td>
      <td>0.038456</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_25_1.png)


To use custom colors, the `custom_palette` flag takes a list of `matplotlib` color names, or RGB tuples.


```python
f,b=bs.contrastplot(df,
                    idx=('Control','Group1'),
                    color_col='Gender',
                    fig_size=(4,6),
                    custom_palette=['skyblue','crimson'] )
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_mannWhitney</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.602739</td>
      <td>0.20855</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.395987</td>
      <td>0.363178</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_27_1.png)


### Floating contrast plots—Multi-plot design
In a multi-plot design, you can horizontally tile two or more two-group floating-contrasts. This is designed to meet data visualization and presentation paradigms that are predominant in academic biomedical research.

This is done mainly through the `idx` option. You can indicate two or more tuples to create a seperate subplot for that contrast.

The effect sizes and confidence intervals for each two-group plot will be computed.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1'),
                         ('Group2','Group3')),
                    color_col='Gender')
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_mannWhitney</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.589500</td>
      <td>0.221113</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.395987</td>
      <td>0.363178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Group2</td>
      <td>Group3</td>
      <td>0.700802</td>
      <td>0.247386</td>
      <td>1.132839</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.002721</td>
      <td>0.002556</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_29_1.png)


### Hub-and-spoke plots

A common experimental design seen in contemporary biomedical research is a shared-control, or 'hub-and-spoke' design. Two or more experimental groups are compared to a common control group.

A hub-and-spoke plot implements estimation statistics and aesthetics on such an experimental design.

If more than 2 columns/groups are indicated in a tuple passed to `idx`, then `contrastplot` will produce a hub-and-spoke plot, where the first group in the tuple is considered the control group. The mean difference and confidence intervals of each subsequent group will be computed against the first control group.


```python
f,b=bs.contrastplot(df,
                    idx=('Control','Group1','Group2','Group3'),
                    color_col='Gender')
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_mannWhitney</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.589396</td>
      <td>0.205610</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.395987</td>
      <td>0.363178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Control</td>
      <td>Group2</td>
      <td>-0.532006</td>
      <td>-0.973154</td>
      <td>-0.084222</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.021575</td>
      <td>0.013580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Control</td>
      <td>Group3</td>
      <td>0.168796</td>
      <td>-0.252951</td>
      <td>0.574814</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.439499</td>
      <td>0.416160</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_32_1.png)


### Hub-and-spoke plots—multi-plot design
You can also horizontally tile two or more hub-and-spoke plots.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1','Group2'),('Group3','Group4'),
                         ('Control','Group5')),
                    color_col='Gender')
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_mannWhitney</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.598425</td>
      <td>0.215760</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>3.959867e-01</td>
      <td>3.631777e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Control</td>
      <td>Group2</td>
      <td>-0.532006</td>
      <td>-0.962146</td>
      <td>-0.085930</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>2.157452e-02</td>
      <td>1.358049e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Group3</td>
      <td>Group4</td>
      <td>3.540697</td>
      <td>3.126591</td>
      <td>3.973550</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>2.518287e-26</td>
      <td>2.424602e-14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Control</td>
      <td>Group5</td>
      <td>-1.397284</td>
      <td>-1.852294</td>
      <td>-0.964113</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>4.761711e-08</td>
      <td>4.267387e-07</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_34_1.png)


## Controlling Aesthetics


```python
# Changing the contrast y-limits.
f,b=bs.contrastplot(df,
                    idx=('Control','Group1','Group2'),
                    color_col='Gender',
                    contrast_ylim=(-2,2))
```


![png](tutorial_img/output_36_0.png)



```python
# Changing the swarmplot y-limits.
f,b=bs.contrastplot(df,
                    idx=('Control','Group1','Group2'),
                    color_col='Gender',
                    swarm_ylim=(-10,10))
```


![png](tutorial_img/output_37_0.png)



```python
# Changing the size of the dots in the swarmplot.
# This is done through swarmplot_kwargs, which accepts a dictionary.
# You can pass any keywords that sns.swarmplot can accept.
f,b=bs.contrastplot(df,
                    idx=('Control','Group1','Group2'),
                    color_col='Gender',
                    swarmplot_kwargs={'size':10}
                   )
```


![png](tutorial_img/output_38_0.png)



```python
# Custom y-axis labels.
f,b=bs.contrastplot(df,
                    idx=('Control','Group1','Group2'),
                    color_col='Gender',
                    swarm_label='My Custom\nSwarm Label',
                    contrast_label='This is the\nContrast Plot'
                   )
```


![png](tutorial_img/output_39_0.png)



```python
# Showing a histogram for the mean summary instead of a horizontal line.
f,b=bs.contrastplot(df,
                    idx=('Control','Group1','Group4'),
                    color_col='Gender',

                    show_means='bars',
                    means_width=0.6 # Changes the width of the summayr bar or the summary line.
                   )
```


![png](tutorial_img/output_40_0.png)


## Appendix: On working with 'melted' DataFrames.

`bs.contrastplot` can also work with 'melted' or 'longform' data. This term is so used because each row will now correspond to a single datapoint, with one column carrying the value (`value`) and other columns carrying 'metadata' describing that datapoint (in this case, `group` and `Gender`).

For more details on wide vs long or 'melted' data, see https://en.wikipedia.org/wiki/Wide_and_narrow_data

To read more about melting a dataframe,see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html


```python
x='group'
y='my_metric'
color_col='Gender'

df_melt=pd.melt(df.reset_index(),
                id_vars=['index',color_col],
                value_vars=cols,value_name=y,var_name=x)

df_melt.head() # Gives the first five rows of `df_melt`.
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Gender</th>
      <th>group</th>
      <th>my_metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>Control</td>
      <td>1.331587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>Control</td>
      <td>0.715279</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Male</td>
      <td>Control</td>
      <td>-1.545400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Male</td>
      <td>Control</td>
      <td>-0.008384</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Male</td>
      <td>Control</td>
      <td>0.621336</td>
    </tr>
  </tbody>
</table>
</div>



If you are using a melted DataFrame, you will need to specify the `x` (containing the categorical group names) and `y` (containing the numerical values for plotting) columns.


```python
f,b=bs.contrastplot(df_melt,
                    x='group',
                    y='my_metric',
                    fig_size=(4,6),
                    idx=('Control','Group1'),
                    color_col='Gender')
b
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference_group</th>
      <th>experimental_group</th>
      <th>stat_summary</th>
      <th>bca_ci_low</th>
      <th>bca_ci_high</th>
      <th>ci</th>
      <th>is_difference</th>
      <th>is_paired</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_mannWhitney</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.583535</td>
      <td>0.237133</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>0.395987</td>
      <td>0.363178</td>
    </tr>
  </tbody>
</table>
</div>




![png](tutorial_img/output_44_1.png)
