from __future__ import division
import numpy as np
import seaborn as sns
from scipy.stats import norm
from collections import OrderedDict
from numpy.random import randint
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, mannwhitneyu, wilcoxon, norm
import warnings

def bootstrap(data, 
              statfunction=None,
              smoothboot=False,
              alpha_level=0.05, 
              reps=3000):
    '''Given a numerical one-dimensional array, 
    the summary statistic and a bootstrapped confidence interval will be returned.

    Keyword arguments:
        data: array, required.
            The data in a one-dimensional array form.

        statfunction: callable, default np.mean
            The summary statistic called on data.

        smoothboot: boolean, default False 
            Taken from seaborn.algorithms.bootstrap.
            If True, performs a smoothed bootstrap (draws samples from a kernel
            destiny estimate).

        alpha: float, default 0.05
            Denotes the likelihood that the confidence interval produced _does not_
            include the true summary statistic. When alpha=0.05, a 95% confidence
            interval is produced.

        reps: int, default 3000
        Number of bootstrap iterations to perform.

    Returns:
        An OrderedDict reporting the summary statistics, percentile CIs,
        bias-corrected and accelerated (BCa) CIs, and the settings used.
    '''
    
    # Taken from scikits.bootstrap code
    # Initialise statfunction
    if statfunction==None:
        statfunction=np.mean
    
    # Compute two-sided alphas.
    if alpha_level>1. or alpha_level <0.:
        raise ValueError("alpha_level must be between 0 and 1.")
    alphas=np.array([alpha_level/2., 1-alpha_level/2.])
    
    # Turns data into array, then tuple.
    data=np.array(data)
    tdata=(data,)

    # The value of the statistic function applied just to the actual data.
    ostat=statfunction(*tdata)
    
    ## Convenience function invoked to get array of desired bootstraps see above!
    # statarray=getstatarray(tdata, statfunction, reps, sort=True)
    statarray=sns.algorithms.bootstrap(data, func=statfunction, n_boot=reps, smooth=smoothboot)
    statarray.sort()

    # Get Percentile indices
    pct_low_high=np.round((reps-1)*alphas)
    pct_low_high=np.nan_to_num(pct_low_high).astype('int')

    # Perform 1-sided t-test and Wilcoxon test; get p-values.
    ttestresult=ttest_1samp(data,0)[1]
    wilcoxonresult=wilcoxon(data)[1]

    # Get Bias-Corrected Accelerated indices convenience function invoked.
    bca_low_high=bca(tdata, alphas, statarray, statfunction, ostat, reps)
    
    # Warnings for unstable or extreme indices.
    for ind in [pct_low_high, bca_low_high]:
        if np.any(ind==0) or np.any(ind==reps-1):
            warnings.warn("Some values used extremal samples results are probably unstable.")
        elif np.any(ind<10) or np.any(ind>=reps-10):
            warnings.warn("Some values used top 10 low/high samples results may be unstable.")
        
    result=OrderedDict()
    result['summary']=ostat
    result['statistic']=str(statfunction)
    result['bootstrap_reps']=reps
    result['ci']=(1-alpha_level)*100
    result['pct_ci_low']=statarray[pct_low_high[0]]
    result['pct_ci_high']=statarray[pct_low_high[1]]
    result['bca_ci_low']=statarray[bca_low_high[0]]
    result['bca_ci_high']=statarray[bca_low_high[1]]
    result['stat_array']=np.array(statarray)
    result['pct_low_high_indices']=pct_low_high
    result['bca_low_high_indices']=bca_low_high
    result['pvalue_1samp_ttest']=ttestresult
    result['pvalue_wilcoxon']=wilcoxonresult
    return result

def bootstrap_contrast(data=None,
                   idx=None,
                   x=None,
                   y=None,
                   statfunction=None,
                   smoothboot=False,
                   alpha_level=0.05, 
                   reps=3000):
    
    # Taken from scikits.bootstrap code
    # Initialise statfunction
    if statfunction==None:
        statfunction=np.nanmean
    # check if idx was parsed
    if idx==None:
        idx=[0,1]
        
    # Compute two-sided alphas.
    alphas=np.array([alpha_level/2, 1-alpha_level/2])
    
    levels=data[x].unique()

    # Two types of dictionaries
    levels_to_idx=dict( zip(list(levels), range(0,len(levels))) ) # levels are the keys.
    idx_to_levels=dict( zip(range(0,len(levels)), list(levels)) ) # level indexes are the keys.
                                                                    # Not sure if I need this latter dict.
    
    # The loop approach below allows us to mix and match level and indices
    # when declaring the idx above.
    arraylist=list() # list to temporarily store the rawdata arrays.
    for i in idx:
        if i in levels_to_idx: # means the supplied id is an actual level
            arraylist.append( np.array(data.ix[data[x]==levels[levels_to_idx[i]]][y]) ) # when I get levels
        elif i in idx_to_levels: # means the supplied id is the level index (does this make sense?)
            arraylist.append( np.array(data.ix[data[x]==levels[i]][y]) ) # when I get level indexes
            
    # Pull out the arrays. 
    # The first array in `arraylist` is the reference array. 
    ref_array=arraylist[0]
    exp_array=arraylist[1]
    
    # Generate statarrays for both arrays.
    ref_statarray=sns.algorithms.bootstrap(ref_array, func=statfunction, n_boot=reps, smooth=smoothboot)
    exp_statarray=sns.algorithms.bootstrap(exp_array, func=statfunction, n_boot=reps, smooth=smoothboot)
    
    diff_array=exp_statarray - ref_statarray
    diff_array_t=(diff_array,) # Note tuple form.
    diff_array.sort()

    # The difference as one would calculate it.
    ostat=statfunction(exp_array) - statfunction(ref_array)
    
    # Get Percentile indices
    pct_low_high=np.round((reps-1)*alphas)
    pct_low_high=np.nan_to_num(pct_low_high).astype('int')

    # Get Bias-Corrected Accelerated indices convenience function invoked.
    bca_low_high=bca(diff_array_t, alphas, diff_array, statfunction, ostat, reps)
    
    # Warnings for unstable or extreme indices.
    for ind in [pct_low_high, bca_low_high]:
        if np.any(ind==0) or np.any(ind==reps-1):
            warnings.warn("Some values used extremal samples results are probably unstable.")
        elif np.any(ind<10) or np.any(ind>=reps-10):
            warnings.warn("Some values used top 10 low/high samples results may be unstable.")
            
    # two-tailed t-test to see if the means of both arrays are different.
    ttestresult=ttest_ind(arraylist[0], arraylist[1])
    
    # Mann-Whitney test to see if the mean of the diff_array is not zero.
    mannwhitneyresult=mannwhitneyu(arraylist[0], arraylist[1])
    
    result=OrderedDict()
    result['summary']=ostat
    result['statistic']=str(statfunction)
    result['bootstrap_reps']=reps
    result['ci']=(1-alpha_level)*100
    result['pct_ci_low']=diff_array[pct_low_high[0]]
    result['pct_ci_high']=diff_array[pct_low_high[1]]
    result['bca_ci_low']=diff_array[bca_low_high[0]]
    result['bca_ci_high']=diff_array[bca_low_high[1]]
    result['diffarray']=np.array(diff_array)
    result['pct_low_high_indices']=pct_low_high
    result['bca_low_high_indices']=bca_low_high
    result['statistic_ref']=statfunction(ref_array)
    result['statistic_exp']=statfunction(exp_array)
    result['ref_input']=arraylist[0]
    result['test_input']=arraylist[1]
    result['pvalue_ttest']=ttestresult[1]
    result['pvalue_mannWhitney']=mannwhitneyresult[1] * 2 # two-sided test result.
    return result

def ci(data, statfunction=np.average, alpha=0.05, n_samples=10000, 
    method='bca', output='lowhigh', epsilon=0.001, multi=None):
    """
This function is taken from C. Evan's code <https://github.com/cgevans/scikits-bootstrap/>
The only modification I have made is to return the array of bootstrapped values, along with
the desired CIs.

Given a set of data ``data``, and a statistics function ``statfunction`` that
applies to that data, computes the bootstrap confidence interval for
``statfunction`` on that data. Data points are assumed to be delineated by
axis 0.
Parameters
----------
data: array_like, shape (N, ...) OR tuple of array_like all with shape (N, ...)
    Input data. Data points are assumed to be delineated by axis 0. Beyond this,
    the shape doesn't matter, so long as ``statfunction`` can be applied to the
    array. If a tuple of array_likes is passed, then samples from each array (along
    axis 0) are passed in order as separate parameters to the statfunction. The
    type of data (single array or tuple of arrays) can be explicitly specified
    by the multi parameter.
statfunction: function (data, weights=(weights, optional)) -> value
    This function should accept samples of data from ``data``. It is applied
    to these samples individually. 
    
    If using the ABC method, the function _must_ accept a named ``weights`` 
    parameter which will be an array_like with weights for each sample, and 
    must return a _weighted_ result. Otherwise this parameter is not used
    or required. Note that numpy's np.average accepts this. (default=np.average)
alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.
n_samples: float, optional
    The number of bootstrap samples to use (default=10000)
method: string, optional
    The method to use: one of 'pi', 'bca', or 'abc' (default='bca')
output: string, optional
    The format of the output. 'lowhigh' gives low and high confidence interval
    values. 'errorbar' gives transposed abs(value-confidence interval value) values
    that are suitable for use with matplotlib's errorbar function. (default='lowhigh')
epsilon: float, optional (only for ABC method)
    The step size for finite difference calculations in the ABC method. Ignored for
    all other methods. (default=0.001)
multi: boolean, optional
    If False, assume data is a single array. If True, assume data is a tuple/other
    iterable of arrays of the same length that should be sampled together. If None,
    decide based on whether the data is an actual tuple. (default=None)
    
Returns
-------
confidences: tuple of floats
    The confidence percentiles specified by alpha
Calculation Methods
-------------------
'pi': Percentile Interval (Efron 13.3)
    The percentile interval method simply returns the 100*alphath bootstrap
    sample's values for the statistic. This is an extremely simple method of 
    confidence interval calculation. However, it has several disadvantages 
    compared to the bias-corrected accelerated method, which is the default.
'bca': Bias-Corrected Accelerated (BCa) Non-Parametric (Efron 14.3) (default)
    This method is much more complex to explain. However, it gives considerably
    better results, and is generally recommended for normal situations. Note
    that in cases where the statistic is smooth, and can be expressed with
    weights, the ABC method will give approximated results much, much faster.
    Note that in a case where the statfunction results in equal output for every
    bootstrap sample, the BCa confidence interval is technically undefined, as
    the acceleration value is undefined. To match the percentile interval method
    and give reasonable output, the implementation of this method returns a
    confidence interval of zero width using the 0th bootstrap sample in this
    case, and warns the user.  
'abc': Approximate Bootstrap Confidence (Efron 14.4, 22.6)
    This method provides approximated bootstrap confidence intervals without
    actually taking bootstrap samples. This requires that the statistic be 
    smooth, and allow for weighting of individual points with a weights=
    parameter (note that np.average allows this). This is _much_ faster
    than all other methods for situations where it can be used.
Examples
--------
To calculate the confidence intervals for the mean of some numbers:
>> boot.ci( np.randn(100), np.average )
Given some data points in arrays x and y calculate the confidence intervals
for all linear regression coefficients simultaneously:
>> boot.ci( (x,y), scipy.stats.linregress )
References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
    """

    # Deal with the alpha values
    if np.iterable(alpha):
        alphas=np.array(alpha)
    else:
        alphas=np.array([alpha/2,1-alpha/2])

    if multi==None:
      if isinstance(data, tuple):
        multi=True
      else:
        multi=False

    # Ensure that the data is actually an array. This isn't nice to pandas,
    # but pandas seems much much slower and the indexes become a problem.
    if multi==False:
      data=np.array(data)
      tdata=(data,)
    else:
      tdata=tuple( np.array(x) for x in data )

    # Deal with ABC *now*, as it doesn't need samples.
    if method=='abc':
        n=tdata[0].shape[0]*1.0
        nn=tdata[0].shape[0]

        I=np.identity(nn)
        ep=epsilon / n*1.0
        p0=np.repeat(1.0/n,nn)

        t1=np.zeros(nn); t2=np.zeros(nn)
        try:
          t0=statfunction(*tdata,weights=p0)
        except TypeError as e:
          raise TypeError("statfunction does not accept correct arguments for ABC ({0})".format(e.message))

        # There MUST be a better way to do this!
        for i in range(0,nn):
            di=I[i] - p0
            tp=statfunction(*tdata,weights=p0+ep*di)
            tm=statfunction(*tdata,weights=p0-ep*di)
            t1[i]=(tp-tm)/(2*ep)
            t2[i]=(tp-2*t0+tm)/ep**2

        sighat=np.sqrt(np.sum(t1**2))/n
        a=(np.sum(t1**3))/(6*n**3*sighat**3)
        delta=t1/(n**2*sighat)
        cq=(statfunction(*tdata,weights=p0+ep*delta)-2*t0+statfunction(*tdata,weights=p0-ep*delta))/(2*sighat*ep**2)
        bhat=np.sum(t2)/(2*n**2)
        curv=bhat/sighat-cq
        z0=norm.ppf(2*norm.cdf(a)*norm.cdf(-curv))
        Z=z0+norm.ppf(alphas)
        za=Z/(1-a*Z)**2
        # stan=t0 + sighat * norm.ppf(alphas)
        abc=np.zeros_like(alphas)
        for i in range(0,len(alphas)):
            abc[i]=statfunction(*tdata,weights=p0+za[i]*delta)

        if output=='lowhigh':
            return abc
        elif output=='errorbar':
            return abs(abc-statfunction(tdata))[np.newaxis].T
        else:
            raise ValueError("Output option {0} is not supported.".format(output))

    # We don't need to generate actual samples; that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes=bootstrap_indexes( tdata[0], n_samples )
    stat=np.array([statfunction(*(x[indexes] for x in tdata)) for indexes in bootindexes])
    stat.sort(axis=0)

    # Percentile Interval Method
    if method=='pi':
        avals=alphas

    # Bias-Corrected Accelerated Method
    elif method=='bca':

        # The value of the statistic function applied just to the actual data.
        ostat=statfunction(*tdata)

        # The bias correction value.
        z0=norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

        # Statistics of the jackknife distribution
        jackindexes=jackknife_indexes(tdata[0])
        jstat=[statfunction(*(x[indexes] for x in tdata)) for indexes in jackindexes]
        jmean=np.mean(jstat,axis=0)

        # Acceleration value
        a=np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )
        if np.any(np.isnan(a)):
            nanind=np.nonzero(np.isnan(a))
            warnings.warn("Some acceleration values were undefined. This is almost certainly because all \
values for the statistic were equal. Affected confidence intervals will have zero width and \
may be inaccurate (indexes: {}). Other warnings are likely related.".format(nanind), InstabilityWarning)
        
        zs=z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

        avals=norm.cdf(z0 + zs/(1-a*zs))

    else:
        raise ValueError("Method {0} is not supported.".format(method))

    nvals=np.round((n_samples-1)*avals)
    
    if np.any(nvals==0) or np.any(nvals==n_samples-1):
        warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
    elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
        warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

    nvals=np.nan_to_num(nvals).astype('int')

    if output=='lowhigh':
        if nvals.ndim==1:
            # All nvals are the same. Simple broadcasting
            return stat[nvals], stat
        else:
            # Nvals are different for each data point. Not simple broadcasting.
            # Each set of nvals along axis 0 corresponds to the data at the same
            # point in other axes.
            return stat[(nvals, np.indices(nvals.shape)[1:].squeeze())], stat

    elif output=='errorbar':
        if nvals.ndim==1:
          return abs(statfunction(data)-stat[nvals])[np.newaxis].T
        else:
          return abs(statfunction(data)-stat[(nvals, np.indices(nvals.shape)[1:])])[np.newaxis].T
    else:
        raise ValueError("Output option {0} is not supported.".format(output))
    
def bootstrap_indexes(data, n_samples=5000):
    """From the scikits.bootstrap package.
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
    base=np.arange(0,len(data))
    return (np.delete(base,i) for i in base)

def getstatarray(tdata, statfunction, reps, sort=True):
    # Convenience function for use within `bootstrap` and `bootstrap_contrast`.
    # Produces `reps` number of bootstrapped samples for `tdata`, using `statfunction`
    # We don't need to generate actual samples that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes=bootstrap_indexes( tdata[0], reps ) # I use the scikits.bootstrap function here.
    statarray=np.array([statfunction(*(x[indexes] for x in tdata)) for indexes in bootindexes])
    if sort is True:
        statarray.sort(axis=0)
    return statarray
        
def bca(data, alphas, statarray, statfunction, ostat, reps):
    '''Subroutine called to calculate the BCa statistics.'''

    # The bias correction value.
    z0=norm.ppf( ( 1.0*np.sum(statarray < ostat, axis=0)  ) / reps )

    # Statistics of the jackknife distribution
    jackindexes=jackknife_indexes(data[0]) # I use the scikits.bootstrap function here.
    jstat=[statfunction(*(x[indexes] for x in data)) for indexes in jackindexes]
    jmean=np.mean(jstat,axis=0)

    # Acceleration value
    a=np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )
    if np.any(np.isnan(a)):
        nanind=np.nonzero(np.isnan(a))
        warnings.warn("Some acceleration values were undefined. \
            This is almost certainly because all values \
            for the statistic were equal. Affected \
            confidence intervals will have zero width and \
            may be inaccurate (indexes: {}). \
            Other warnings are likely related.".format(nanind))
    zs=z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals=norm.cdf(z0 + zs/(1-a*zs))
    nvals=np.round((reps-1)*avals)
    nvals=np.nan_to_num(nvals).astype('int')
    
    return nvals