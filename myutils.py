''' A module with a bunch of different utility functions. '''

from matplotlib.mlab import find
import numpy as np
from scipy.stats import mannwhitneyu
from numpy.random import *


def ranksum(samp1, samp2):
    ''' Calculates the U statistic and probability that the samples are from
    two different distributions.  These tests are non-parametric, so you can
    use them if your sample distributions are not Gaussian.  The null hypothesis
    for the test is that the two samples come from the same distribution, so
    if p is less than some cutoff, you can reject the null hypothesis and claim
    the samples come from different distributions, that is, one sample is ranked
    higher (for instance, larger times, higher spiking rates) than the other sample.
    
    For small sample sizes (n, m <30), the U statistic is calculated directly.
    The probability is found from a table, at the p<0.05 level.
    
    For large sample sizes (n, m >30), the U statistic and probability are
    calculated using scipy.stats.mannwhitneyu which uses a normal approximation.
    
    Parameters
    ----------
    samp1 : array like
        A 1D array of the sample data
    samp2 : array like
        A 1D array of the sample data
        
    Returns
    -------
    U : int
        The smaller U statistic of the sample
    p : float
        The probability that the null hypothesis is true.
    '''
    
    if (len(samp1) <= 30) & (len(samp2) <= 30):
        return ranksum_small(samp1, samp2)
    else:
        return mannwhitneyu(samp1, samp2)

def ranksum_small(samp1, samp2):
    ''' This function tests the null hypothesis that two related samples come
    from the same distribution.  This function is for sample sizes < 20, 
    otherwise use scipy.stats.mannwhitneyu or scipy.stats.ranksums, etc.
    
    Parameters
    ----------
    samp1 : array like
        A 1D array of the sample data
    samp2 : array like
        A 1D array of the sample data
        
    Returns
    -------
    U : int
        The smaller U statistic of the sample
    p : float
        The probability that the null hypothesis is true.  Since p is
        found from a lookup table, p = 0.05 if U is smaller than the
        critical value, p = 1 otherwise.
        
    '''
    
    #~ if np.mean(samp1) > np.mean(samp2):
        #~ # Do nothing
        #~ s1 = samp1
        #~ s2 = samp2
        #~ pass
    #~ elif np.mean(samp1) < np.mean(samp2):
        #~ # Switch them
        #~ s1 = samp2
        #~ s2 = samp1
    s1 = samp1
    s2 = samp2
    
    # Create a struct array to store sample values and labels
    dt = np.dtype([('value', 'f8'), ('sample', 'i8')])
    ranking = np.zeros((len(s1) + len(s2),), dtype = dt)
    
    # Fill the struct array
    ranking['value'] = np.concatenate((s1,s2))
    ranking['sample'] = np.concatenate((np.ones(len(s1)), 
        np.ones(len(s2))*2 )).astype(int)
    
    # Sort by value to order by rank
    ranking.sort(order='value')
    ranking = ranking[::-1]
    ones = find(ranking['sample'] == 1)
    twos = find(ranking['sample'] == 2)
    
    # Need to randomize ordering of zeros
    zero_ind = find(ranking['value']==0)
    ranking[zero_ind] = permutation(ranking[zero_ind])
    
    # Calculate the U statistic for the first distribution
    ranksums1 = []
    for ii in ones:
        smaller2 = find(ranking['sample'][:ii]==2)
        for jj in smaller2:
            if ranking['value'][ii] == ranking['value'][jj]:
                ranksums1.append(0.5)
            else:
                ranksums1.append(1)
    U1 = np.sum(ranksums1)
    
    # Calculate the U statistic for the second distribution
    ranksums2 = []
    for ii in twos:
        smaller1 = find(ranking['sample'][:ii]==1)
        for jj in smaller1:
            if ranking['value'][ii] == ranking['value'][jj]:
                ranksums2.append(0.5)
            else:
                ranksums2.append(1)
    U2 = np.sum(ranksums2)
    
    # Check significance
    
    if len(s1) <= len(s2):
        sig1 = U1 < _crit_u(len(s1),len(s2))
        sig2 = U2 < _crit_u(len(s1),len(s2))
    elif len(s1) > len(s2):
        sig1 = U1 < _crit_u(len(s2),len(s1))
        sig2 = U2 < _crit_u(len(s2),len(s1))
    
    if (sig1) | (sig2):
        p = 0.05
    else:
        p = 1
    
    return np.min([U1, U2]), p

def _crit_u(size1, size2):
    ''' This is basically just a table of critical U values for p < 0.05 '''
    
    if (size1 < 3) | (size1 > 30):
        raise ValueError, 'size1 must be between 3 and 30, inclusive'
    
    if (size2 < 5) | (size1 > 30):
        raise ValueError, 'size2 must be between 5 and 30, inclusive'
    
    crits = np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13],
        [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 23],
        [2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33],
        [0, 5, 6, 8, 10, 11, 13, 14, 16, 17, 19, 21, 22, 24, 25, 27, 29, 30, 32, 33, 35, 37, 38, 40, 42, 43],
        [0,0,8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54],
        [0,0,0,13, 15, 17, 19, 22, 24, 26, 29, 31, 34, 36, 38, 41, 43, 45, 48, 50, 53, 55, 57, 60, 62, 65],
        [0,0,0,0,17, 20, 23, 26, 28, 31, 34, 37, 39, 42, 45, 48, 50, 53, 56, 59, 62, 64, 67, 70, 73, 76],
        [0,0,0,0,0,23, 26, 29, 33, 36, 39, 42, 45, 48, 52, 55, 58, 61, 64, 67, 71, 74, 77, 80, 83, 87],
        [0,0,0,0,0,0,30, 33, 37, 40, 44, 47, 51, 55, 58, 62, 65, 69, 73, 76, 80, 83, 87, 90, 94, 98],
        [0,0,0,0,0,0,0,37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109],
        [0,0,0,0,0,0,0,0,45, 50, 54, 59, 63, 67, 72, 76, 80, 85, 89, 94, 98, 102, 107, 111, 116, 120],
        [0,0,0,0,0,0,0,0,0,55, 59, 64, 67, 74, 78, 83, 88, 93, 98, 102, 107, 112, 118, 122, 127, 131],
        [0,0,0,0,0,0,0,0,0,0,64, 70, 75, 80, 85, 90, 96, 101, 106, 111, 117, 122, 125, 132, 138, 143],
        [0,0,0,0,0,0,0,0,0,0,0,75, 81, 86, 92, 98, 103, 109, 115, 120, 126, 132, 138, 143, 149, 154],
        [0,0,0,0,0,0,0,0,0,0,0,0,87, 93, 99, 105, 111, 117, 123, 129, 135, 141, 147, 154, 160, 166],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,99, 106, 112, 119, 125, 132, 138, 145, 151, 158, 164, 171, 177],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,113, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127, 134, 141, 149, 156, 163, 171, 178, 186, 193, 200],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142, 150, 157, 165, 173, 181, 188, 196, 204, 212],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,158, 166, 174, 182, 191, 199, 207, 215, 223],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,175, 183, 192, 200, 209, 218, 226, 235],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192, 201, 210, 219, 228, 238, 247],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,211, 220, 230, 239, 249, 258],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,230, 240, 250, 260, 270],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,250, 261, 271, 282],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,272, 282, 293],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,294, 305],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,317]]).astype(int)
    
    return crits[size1-3,size2-5]

def multinormal(x, mean, cov):
    
    dims = len(mean)
    dcov = np.linalg.det(cov)
    icov = np.linalg.inv(cov)
    dif = x - mean
    expo = np.exp(-0.5 * np.dot(np.dot(dif, icov), dif))
    norm = (2* np.sqrt( np.power(np.pi, dims) * dcov))
    
    return (expo / norm)

def normal(x, mean, sig):
    
    expo = np.exp(-0.5*(x-mean)**2/sig**2)
    norm = np.sqrt(2*np.pi*sig**2)
    
    return (expo / norm)
    
def kurtosis(data):
    
    mu = np.mean(data)
    pdf, x = np.histogram(data, bins = 100, range = (0,100), density = True)
    
    x = x[:-1] + 0.5
    # Subtracting 3 for excess kurtosis
    try:
        kurt = np.sum((x-mu)**4*pdf)/np.std(data)**4 - 3
        return kurt
    except:
        pass
    

def kurtosis2(data):
    
    mu = np.mean(data)
    n = float(len(data))
    numer = sum((data-mu)**4)/n
    denom = (sum((data-mu)**2)/n)**2
    try:
        kurt = numer/denom - 3
        return kurt
    except:
        pass
    
    
def bootstrap(data, measure_func, iters = 10000):
    ''' Compute the distribution of a statistic (mean, variance, etc.)
        using the bootstrapping method, sampling with replacement.
        
        Parameters
        ----------
        
        data : array like
            The input sampled data.
        measure_func : function
            The function for the statistic that will be calculated from
            bootstrap samples.  For instance, if you want to calculate the
            distribution of the means of the sample, use np.mean for
            measure_func.
        iters : int
            The number of bootstrapping samples you want to get from your
            data.  Higher is better, default is 10000.
            
        Returns
        -------
        out : np.array
            The output of measure_func for each of the bootstrap samples.
        
    '''
    
    out = map(measure_func, BootSample(data, iters))
    
    return np.array(out)
    
class BootSample(object):
    ''' A generator used for bootstrapping.  Samples with replacement from
        a data array.
        
        Parameters
        ----------
       
        iters : int
            The number of iterations before the generator runs out.
        
        Returns
        -------
        out : np.array
            Yields an np.array of pseudorandom integers.  The array
            is max integers long, the integers are between min and max.
    
    '''
    
    def __init__(self, data, iters):
        self._data = data
        self._iters = iters
        
    def __iter__(self):
        for ii in np.arange(self._iters):
            sample = randint(0,len(self._data),len(self._data))
            if hasattr(self._data, 'ix'):
                index = self._data.index
                samp = self._data.ix[index[sample]]
            else:
                samp = self._data[sample]
            yield samp
    
def calculate_rates(all_units):
    
    import os
    import cPickle as pkl
    
    sessions = np.unique([ unit.session for unit in all_units ])

    for session in sessions:

        units = session.units
        tetrodes = np.unique([ unit.tetrode for unit in units ])
        path = os.path.expanduser(session.path)
        
        for tetrode in tetrodes:
            # Load the data
            filename = '%s_%s.cls.%s' % (session.rat, session.date, tetrode)
            path_filename = os.path.join(path, filename)
            with open(path_filename,'r') as cls:
                clusters = pkl.load(cls)
            
            units = [ unit for unit in session.units if unit.tetrode == tetrode ]
            
            for unit in units:
                unit_data = clusters[unit.cluster]
                if type(unit_data['peaks']) == type(np.array(1)):
                    spike_count = len(unit_data['peaks'])
                else:
                    spike_count = 0
                unit.rate = spike_count / session.duration