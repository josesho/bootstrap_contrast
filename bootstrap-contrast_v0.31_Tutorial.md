
## Load Libraries


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn.apionly as sns
import bootstrap_contrast as bs 
# note this is different from `bootstrapContrast` in versions prior to v0.30.

import pandas as pd
import numpy as np
import scipy as sp
```

## Create dummy dataset

Here, we create a dummy dataset to illustrate how `bootstrap-contrast` functions.
We first create it as a 'wide' dataset, where each column corresponds to a group of observations, and each row is simply an index number referring to an observation.


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




    <bootstrap_contrast.bootstrap_tools.bootstrap at 0x118461cc0>



It has several callable attributes. Of interest is its `results` attribute, which returns a dictionary summarising the results of the contrast computation.


```python
contr.results
```




    {'bca_ci_high': 0.24548837217502079,
     'bca_ci_low': -0.5877837620341424,
     'ci': 95.0,
     'is_difference': True,
     'is_paired': False,
     'stat_summary': -0.18080446527038216}



`is_paired` indicates the two arrays are paired (or repeated) observations. This is indicated by the `paired` flag.


```python
contr_paired=bs.bootstrap(df['Control'],df['Group1'],
                         paired=True)
contr_paired.results
```




    {'bca_ci_high': 0.2315358954772489,
     'bca_ci_low': -0.56709424589376278,
     'ci': 95.0,
     'is_difference': True,
     'is_paired': True,
     'stat_summary': -0.1808044652703821}



`is_difference` basically indicates if one or two arrays were passed to the `bootstrap` function. Obseve what happens if we just give one array.


```python
just_control_=bs.bootstrap(df['Control'])
just_control_.results
```




    {'bca_ci_high': 0.48092618004349924,
     'bca_ci_low': -0.12687439021613289,
     'ci': 95.0,
     'is_difference': False,
     'is_paired': False,
     'stat_summary': 0.1717562151007304}



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

### Gardner-Altman floating contrast plots

Below we produce three aligned Gardner-Altman floating contrast plots. 

The `contrastplot` command will return 2 objects: a matplotlib `Figure` and a pandas `DataFrame`.
In the Jupyter Notebook, with `%matplotlib inline`, the figure should automatically appear.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
    color_col='Gender',
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_21_1.png)


The pandas `DataFrame` returned by `bs.bootstrap` contains all the (pairwise) comparisons made in the course of generating the plot, with confidence intervals (95% by default) and relevant p-values.


```python
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
      <th>pvalue_1samp_ttest</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_2samp_paired_ttest</th>
      <th>pvalue_mannWhitney</th>
      <th>pvalue_wilcoxon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.583302</td>
      <td>0.223463</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>NIL</td>
      <td>3.959867e-01</td>
      <td>NIL</td>
      <td>3.631777e-01</td>
      <td>NIL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Control</td>
      <td>Group3</td>
      <td>0.168796</td>
      <td>-0.244889</td>
      <td>0.586373</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>NIL</td>
      <td>4.394990e-01</td>
      <td>NIL</td>
      <td>4.161598e-01</td>
      <td>NIL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Control</td>
      <td>Group5</td>
      <td>-1.397284</td>
      <td>-1.830624</td>
      <td>-0.934402</td>
      <td>95.0</td>
      <td>True</td>
      <td>False</td>
      <td>NIL</td>
      <td>4.761711e-08</td>
      <td>NIL</td>
      <td>4.267387e-07</td>
      <td>NIL</td>
    </tr>
  </tbody>
</table>
</div>



`bs.bootstrap` will automatically drop any NaNs in the data. 


```python
# make a copy of the data
df2=df.copy()
# add nans randomly
for j,c in enumerate(cols):
    # get a random number of rows to set as NaNs
    np.random.seed(20+j)
    numNan=np.random.randint(10) 
    # get indices (number=numNan) to set as NaNs.
    np.random.seed(21+j)
    idx=np.random.randint(0,39,numNan)
    df2.loc[idx,c]=np.nan
```


```python
df2
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
      <th>Control</th>
      <th>Group1</th>
      <th>Group2</th>
      <th>Group3</th>
      <th>Group4</th>
      <th>Group5</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.331587</td>
      <td>NaN</td>
      <td>0.372986</td>
      <td>NaN</td>
      <td>5.706473</td>
      <td>-1.343561</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.715279</td>
      <td>-0.286073</td>
      <td>-0.781426</td>
      <td>0.953766</td>
      <td>4.087105</td>
      <td>NaN</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.545400</td>
      <td>-0.484565</td>
      <td>0.142439</td>
      <td>0.155497</td>
      <td>4.191374</td>
      <td>-1.171499</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.008384</td>
      <td>-2.653319</td>
      <td>-1.800736</td>
      <td>NaN</td>
      <td>3.920430</td>
      <td>-1.551969</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.653143</td>
      <td>1.545102</td>
      <td>NaN</td>
      <td>-0.740874</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.720086</td>
      <td>-0.319631</td>
      <td>-1.634721</td>
      <td>0.732338</td>
      <td>4.159146</td>
      <td>-2.939966</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.265512</td>
      <td>-0.536629</td>
      <td>-0.094873</td>
      <td>1.550188</td>
      <td>2.348715</td>
      <td>NaN</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.108549</td>
      <td>0.315403</td>
      <td>-0.220228</td>
      <td>1.061211</td>
      <td>4.232220</td>
      <td>-2.196542</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.004291</td>
      <td>NaN</td>
      <td>-0.906982</td>
      <td>1.678686</td>
      <td>3.385974</td>
      <td>-1.335687</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>-1.065603</td>
      <td>NaN</td>
      <td>-0.845377</td>
      <td>5.192982</td>
      <td>-1.521123</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.433026</td>
      <td>-0.886240</td>
      <td>-0.697823</td>
      <td>-0.588989</td>
      <td>3.795082</td>
      <td>-1.220654</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.203037</td>
      <td>-0.475733</td>
      <td>0.372457</td>
      <td>-1.061606</td>
      <td>4.016128</td>
      <td>-0.609284</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.965066</td>
      <td>0.689682</td>
      <td>NaN</td>
      <td>0.762847</td>
      <td>2.816874</td>
      <td>-0.241531</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.028274</td>
      <td>0.561192</td>
      <td>-1.315169</td>
      <td>-0.043326</td>
      <td>4.706477</td>
      <td>-0.548351</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.228630</td>
      <td>-1.305549</td>
      <td>1.242356</td>
      <td>1.113741</td>
      <td>3.801630</td>
      <td>-1.621476</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NaN</td>
      <td>-1.119475</td>
      <td>-0.222150</td>
      <td>0.517351</td>
      <td>4.682330</td>
      <td>-0.340670</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-1.136602</td>
      <td>0.736837</td>
      <td>0.912515</td>
      <td>0.327303</td>
      <td>4.892072</td>
      <td>-1.179230</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.135137</td>
      <td>1.574634</td>
      <td>-1.013869</td>
      <td>2.350383</td>
      <td>4.855729</td>
      <td>0.760236</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.484537</td>
      <td>-0.031075</td>
      <td>-1.129530</td>
      <td>0.806289</td>
      <td>3.738761</td>
      <td>-0.250210</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-1.079805</td>
      <td>-0.683447</td>
      <td>NaN</td>
      <td>0.173228</td>
      <td>1.918896</td>
      <td>NaN</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-1.977728</td>
      <td>NaN</td>
      <td>0.401872</td>
      <td>-0.784161</td>
      <td>2.710666</td>
      <td>-1.096558</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-1.743372</td>
      <td>-0.309577</td>
      <td>0.038846</td>
      <td>1.390705</td>
      <td>4.919828</td>
      <td>-2.080330</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.266070</td>
      <td>0.725752</td>
      <td>0.540761</td>
      <td>1.152831</td>
      <td>5.110201</td>
      <td>-0.866140</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.384967</td>
      <td>1.549072</td>
      <td>0.427333</td>
      <td>-0.887182</td>
      <td>5.422409</td>
      <td>-2.251181</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.123691</td>
      <td>0.630080</td>
      <td>-1.254360</td>
      <td>0.054789</td>
      <td>3.395736</td>
      <td>-0.616097</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.672622</td>
      <td>0.073493</td>
      <td>-2.313333</td>
      <td>0.437858</td>
      <td>2.920116</td>
      <td>-3.044364</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.099149</td>
      <td>0.732271</td>
      <td>-1.781757</td>
      <td>-1.439093</td>
      <td>NaN</td>
      <td>-2.283900</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.397996</td>
      <td>-0.642575</td>
      <td>-1.888094</td>
      <td>-0.078135</td>
      <td>4.960377</td>
      <td>0.567387</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.271248</td>
      <td>-0.178093</td>
      <td>-2.318535</td>
      <td>1.599238</td>
      <td>4.024322</td>
      <td>0.646222</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.613204</td>
      <td>NaN</td>
      <td>-0.747431</td>
      <td>-1.415108</td>
      <td>3.995442</td>
      <td>0.418925</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>30</th>
      <td>-0.267317</td>
      <td>-0.204375</td>
      <td>-0.628404</td>
      <td>0.690872</td>
      <td>2.515961</td>
      <td>-2.992920</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-0.549309</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.092742</td>
      <td>3.770960</td>
      <td>-2.648138</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.132708</td>
      <td>-0.185775</td>
      <td>0.114976</td>
      <td>-0.420980</td>
      <td>5.459194</td>
      <td>-2.595158</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-0.476142</td>
      <td>-0.380536</td>
      <td>-0.484359</td>
      <td>-0.253752</td>
      <td>2.992113</td>
      <td>-2.863298</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1.308473</td>
      <td>0.088978</td>
      <td>-0.353904</td>
      <td>NaN</td>
      <td>3.482986</td>
      <td>-0.750010</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.195013</td>
      <td>0.063672</td>
      <td>-0.026748</td>
      <td>0.714329</td>
      <td>3.835613</td>
      <td>-1.538708</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.400210</td>
      <td>NaN</td>
      <td>-1.097204</td>
      <td>0.597241</td>
      <td>3.642214</td>
      <td>-1.000581</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>37</th>
      <td>-0.337632</td>
      <td>1.402771</td>
      <td>-0.813856</td>
      <td>-1.312845</td>
      <td>2.013704</td>
      <td>-1.539278</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1.256472</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.564034</td>
      <td>3.449593</td>
      <td>NaN</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>39</th>
      <td>-0.731970</td>
      <td>1.295619</td>
      <td>-0.777945</td>
      <td>0.301270</td>
      <td>3.378741</td>
      <td>1.253789</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>




```python
f,b=bs.contrastplot(df2,
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
    color_col='Gender',
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_27_1.png)


Note how the Ns (appended to the group names in the xtick labels) indicate the number of datapoints being plotted, and used to calculate the contrasts.


We can also display paired data.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
                    color_col='Gender',
                    paired=True
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_29_1.png)


If you want to plot the raw swarmplot instead of the paired lines, use the `show_pairs` flag to set this.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
                    color_col='Gender',
                    paired=True,
                    show_pairs=False
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_31_1.png)


The contrasts computed will still be paired, as indicated by the DataFrame produced.


```python
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
      <th>pvalue_1samp_ttest</th>
      <th>pvalue_2samp_ind_ttest</th>
      <th>pvalue_2samp_paired_ttest</th>
      <th>pvalue_mannWhitney</th>
      <th>pvalue_wilcoxon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Control</td>
      <td>Group1</td>
      <td>-0.180804</td>
      <td>-0.570014</td>
      <td>0.226075</td>
      <td>95.0</td>
      <td>True</td>
      <td>True</td>
      <td>NIL</td>
      <td>NIL</td>
      <td>3.931001e-01</td>
      <td>NIL</td>
      <td>0.353693</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Control</td>
      <td>Group3</td>
      <td>0.168796</td>
      <td>-0.290572</td>
      <td>0.610817</td>
      <td>95.0</td>
      <td>True</td>
      <td>True</td>
      <td>NIL</td>
      <td>NIL</td>
      <td>4.786384e-01</td>
      <td>NIL</td>
      <td>0.510138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Control</td>
      <td>Group5</td>
      <td>-1.397284</td>
      <td>-1.834472</td>
      <td>-0.960863</td>
      <td>95.0</td>
      <td>True</td>
      <td>True</td>
      <td>NIL</td>
      <td>NIL</td>
      <td>2.114260e-07</td>
      <td>NIL</td>
      <td>0.000004</td>
    </tr>
  </tbody>
</table>
</div>



To use custom colors, the `custom_palette` flag takes a list of `matplotlib` color names, or RGB tuples.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
                    color_col='Gender',
                    custom_palette=['skyblue','crimson']
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_35_1.png)


`bs.contrastplot` can also work with 'melted' or 'longform' data. This term is so used because each row will now correspond to a single datapoint, with one column carrying the value (`value`) and other columns carrying 'metadata' describing that datapoint (in this case, `group` and `Gender`).

For more details on wide vs long or 'melted' data, see
https://en.wikipedia.org/wiki/Wide_and_narrow_data


```python
x='group'
y='some_metric'
color_col='Gender'
# To read more about melting a dataframe,
# see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html
df_melt=pd.melt(df.reset_index(),
                id_vars=['index',color_col],
                value_vars=cols,value_name=y,var_name=x)
```

If you are using a melted DataFrame, you will need to specify the `x` (containing the categorical group names) and `y` (containing the numerical values for plotting) columns.


```python
f,b=bs.contrastplot(df_melt,
                    x='group',
                    y='some_metric',
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
                    color_col='Gender',
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_39_1.png)


### Cummings contrast and hub-and-spoke plots.

A Cummings contrast plot.


```python
f,b=bs.contrastplot(df,
                    idx=(('Control','Group1',),('Control','Group3'),('Control','Group5')),
                    color_col='Gender',
                    float_contrast=False
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_42_1.png)


A Cummings hub-and-spoke contrast plot, where all groups are compared relative to the very first category in each tuple.


```python
f,b=bs.contrastplot(df2,
                    idx=(('Control','Group1','Group2'),('Group3','Group4'),
                         ('Control','Group5')),
                    color_col='Gender',
)
```

    /Users/josesho/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



![png](bootstrap-contrast_v0.31_Tutorial_files/bootstrap-contrast_v0.31_Tutorial_44_1.png)



```python

```
