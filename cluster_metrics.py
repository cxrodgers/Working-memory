"""  A module containing functions that calculate metrics for testing the 
quality of clustered data.

Load data from KlustaKwik files using the load_spikes() function.
Get false positive and false negative estimates using metrics()

Author:  Mat Leonard
Last modified: 8/2/2012
"""

import numpy as np

from scipy.special import erf
try:
    from sklearn import mixture
except ImportError:
    from scikits.learn import mixture
from itertools import combinations
try:
    from KKFileSchema import KKFileSchema
except ImportError:
    from kkpandas.KKFileSchema import KKFileSchema
import os
import kkio
#~ try:
    #~ import kkio
#~ except:
    #~ from kkpandas import kkio


def load_spikes(data_dir, group, samp_rate, n_samp, n_chan):
    ''' This function takes the feature, cluster, and spike files in KlustaKwik format 
    and pulls out the features, spike times, spike waveforms for each cluster.
   
    Parameters
    -----------------------------------------
    data_dir : path to the directory with the KlustaKwik files
    group : the group number you want to load
    samp_rate : the sampling rate of the recording in samples per second
    n_samp : number of samples for each stored spike in the spike file
    n_chan : number of channels stored in the spike file
    
    Returns
    -----------------------------------------
    out : dict
        out['features'] : dictionary of clustered features
        out['times'] :  dictionary of clustered spike times
        out['waveforms'] : dictionary of clustered spike waveforms
    '''
    # Get the clustered data from Klustakwik files
    kfs = KKFileSchema.coerce(data_dir)
    
    # Get the features and spike time stamps
    feat = kkio.read_fetfile(kfs.fetfiles[group])
    features = feat.values[:,:-1]
    time_stamps = feat.time.values
    
    # Get spike cluster labels
    clu = kkio.read_clufile(kfs.clufiles[group])
    
    # Get the spike waveforms
    spikes = kkio.read_spkfile(kfs.spkfiles[group])
    
    # Reshape the spike waveforms into a useful form
    spikes = spikes.reshape((len(spikes)/(n_chan*n_samp), (n_chan*n_samp)))
    for ii, spike in enumerate(spikes):
        spikes[ii] = spike.reshape((n_chan, n_samp), order = 'F').reshape(n_chan*n_samp)
    
    # Convert spike waveforms into voltage
    spikes = spikes*(8192.0/2.**16)
    
    # Cluster numbers
    cluster_nums = np.unique(clu.values)
    
    # Grouping the indices by cluster
    cluster_ind = [ np.nonzero(clu.values == n)[0] for n in cluster_nums ]
   
    # Get the spike times for each cluster
    times = [ time_stamps[ind]/np.float(samp_rate) for ind in cluster_ind ]
    
    # Get the features for each cluster
    feats = [ features[ind] for ind in cluster_ind ]
    
    # Get the spike waveforms for each cluster
    spks = [ spikes[ind] for ind in cluster_ind ]
    
    # Make a dictionary where each key is the cluster number and the value
    # is an array of the spike times in that cluster
    clustered_times = dict(zip(cluster_nums, times))
    
    # Make a dictionary where each key is the cluster number and the value
    # is an array of the features in that cluster
    clustered_features = dict(zip(cluster_nums, feats))
    
    # Make a dictionary where each key is the cluster number and the value
    # is an array of the features in that cluster
    clustered_waveforms = dict(zip(cluster_nums, spks))
    
    # Let's make sure the spike times for each cluster are sorted correctly
    for spikes in clustered_times.itervalues():
        spikes.sort()
    
    out_dict = {'features' : clustered_features, 'times' : clustered_times,
        'waveforms' : clustered_waveforms }
    
    return out_dict

def refractory(clustered_times, t_ref, t_cen, t_exp):
    ''' Returns an estimation of false positives in a cluster based on the
    number of refractory period violations.  Returns NaN if an estimate can't
    be made, or if the false positive estimate is greater that 0.5.  This is 
    most likely because of too many spikes in the refractory period.
    
    Parameters
    -----------------------------------------
    clustered_times : dictionary of clusters from the load_spikes function
    t_ref : the width of the refractory period, in seconds
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    
    This algorithm follows from D.N. Hill, et al., J. Neuroscience, 2011
    '''
   
    
    # Make a dictionary to store the false positive rates
    f_p_1 = dict.fromkeys(clustered_times.keys())
    
    # Calculate the false positive rate for each cluster.
    for clst, times in clustered_times.iteritems():
        # This finds the time difference between consecutive peaks
        isi = np.array([ jj - ii for ii, jj in zip(times[:-1], times[1:]) ])
        
        # Now we need to find the number of refractory period violations
        ref = np.sum(isi <= t_ref)
        
        # The number of spikes in the cluster
        N = float(len(times))

        # If there are no spikes in the refractory period, then set the
        # false positive rate to 0
        if ref == 0:
            f_p_1[clst] = 0
    
        cons = t_exp/(t_ref - t_cen)/2./N/N
        
        f_p_1[clst] =(1 - np.sqrt(1-4*ref*cons))/2.
    
    return f_p_1

def threshold(clustered_waveforms, thresh):
    ''' Returns the rate of false negatives caused by spikes below the 
    detection threshold
    
    Parameters
    -----------------------------------------
    clustered_waveforms : dictionary of clustered waveforms, you get this 
        from load_spikes function
    thresh : detection threshold used for spike sorting
    
    Returns
    -----------------------------------------
    out : dictionary
    f_n : false negative percentages for each cluster
    '''
    
    # We need to get a histogram of the spike heights
    
    cl_spikes = clustered_waveforms
    
    # Create dictionary to store the false negative values
    f_n = dict.fromkeys(cl_spikes.keys())
    
    for clst, spikes in cl_spikes.iteritems():
        # Get the peak heights for each spike
        peaks = [ np.min(spk) for spk in spikes ]
        
        freq, x = np.histogram( peaks, 50 )
        
        X = x[:-1] + (x[1]-x[0])/2.
        
        # Find the moments of the data distribution
        mu = np.sum(X*freq)/np.sum(freq)
        sigma = np.sqrt(np.abs(np.sum((X-mu)**2*freq)/np.sum(freq)))
        
        # The false negative percentage is the cumulative Gaussian function
        # up to the detection threshold
        
        cdf = 0.5*(1 + erf((thresh - mu)/np.sqrt(2*sigma**2)))
        f_n[clst] = (1-cdf) 
            
    return f_n
    
def overlap(clustered_features, ignore = [0]):
    ''' Okay, so we are going to calculate the false positives and negatives
    due to overlap between clusters.
    
    This function is a little untrustworthy unless the clusters are pretty
    well behaved.  It calculates the false positives and negatives by
    fitting a gaussian mixture model with two classes to each cluster
    pair.  Then the false positives and negatives between those two clusters
    are calculated.  If you have very non-gaussian shaped clusters, this 
    probably won't work very well.  The work around will be to ignore these
    bad clusters.
    
    Parameters
    -----------------------------------------
    ignore: a list of the clusters to ignore in the analysis.  Default is cluster
    zero since that is typically the noise cluster.
    '''
    
    c_feat = clustered_features
    
    # Make a dictionary to store the false positives
    f_p = dict.fromkeys(c_feat.keys())
    
    # Make a dictionary to store the false negatives
    f_n = dict.fromkeys(c_feat.keys())
    
    # This part is going to take out clusters we want to ignore
    keys = c_feat.keys()
    [ keys.pop(keys.index(ig)) for ig in ignore if ig in keys ]
    
    # We're going to need to fit models to all pairwise combinations
    # If the first cluster is a noise cluster, ignore it
    comb_pairs = combinations(keys, 2)
    
    # Fit the models, two clusters at a time
    for clst_1, clst_2 in comb_pairs:
        # Make the GMM
        gmm = mixture.GMM(n_components = 2, covariance_type = 'full', 
            init_params='')
        
        # The data we are going to fit the model to
        data_vecs = np.concatenate((c_feat[clst_1], c_feat[clst_2]))
        
        # Fit the GMM to the data
        gmm.fit(data_vecs)
        
        # If the model hasn't converged, keep training it
        while gmm.converged_ == False:
            gmm.fit(data_vecs)
        
        # Calculate the false positives and negatives for each cluster
        # For each pair, there is one model, but four values we can get
        k = clst_1
        i = clst_2
        N_k = np.float(len(c_feat[k]))
        N_i = np.float(len(c_feat[i]))
        
        # This calculates the average probability (over k) that a spike in 
        # cluster k belongs to cluster i - false positives
        f_p_k_i = 1/N_k*np.min(np.sum(gmm.predict_proba(c_feat[k]), axis=0))
        
        # This calculates the average probability (over k)  that a spike in 
        # cluster i belongs to cluster k - false negatives
        f_n_k_i = 1/N_k*np.min(np.sum(gmm.predict_proba(c_feat[i]), axis=0))
        
        # This calculates the average probability (over i) that a spike in 
        # cluster i belongs to cluster k - false positives
        f_p_i_k = 1/N_i*np.min(np.sum(gmm.predict_proba(c_feat[i]), axis=0))
        
        # This calculates the average probability (over i) that a spike in 
        # cluster k belongs to cluster i - false negatives
        f_n_i_k = 1/N_i*np.min(np.sum(gmm.predict_proba(c_feat[k]), axis=0))
        
        # Now store these values
        # Probably a better way to do this but I haven't put brain power to it
        if type(f_p[k]) != list:
            f_p[k] = [f_p_k_i]
        else:
            f_p[k].append(f_p_k_i)
        
        if type(f_p[i]) != list:
            f_p[i] = [f_p_i_k]
        else:
            f_p[i].append(f_p_i_k)
        
        if type(f_n[k]) != list:
            f_n[k] = [f_n_k_i]
        else:
            f_n[k].append(f_n_k_i)
        
        if type(f_n[i]) != list:
            f_n[i] = [f_n_i_k]
        else:
            f_n[i].append(f_n_i_k)
    
    # Sum over all values from pairs of clusters
    for clst in f_p.iterkeys():
        f_p[clst] = np.sum(f_p[clst])
        f_n[clst] = np.sum(f_n[clst])
        
    return f_p, f_n

def censored(clustered_times, t_cen, t_exp):
    ''' Returns the estimated false negative rate caused by spikes censored
    after a detected spike
    
    Parameters
    -----------------------------------------
    clustered_times : dictionary of clusters from the load_spikes function
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    '''
    
    # Get the total number of spikes first
    N = np.sum([len(times) for times in clustered_times.itervalues()])
    
    # Create dictionary to store the false negative values
    f_n = dict.fromkeys(clustered_times.keys())
    
    for clst, times in clustered_times.iteritems():
        f_n[clst] = (N - len(times)) * t_cen / t_exp
    
    return f_n

def metrics(data, thresh, t_ref, t_cen, t_exp, ignore = [0]):
    ''' This function runs all the metrics calculations and sums them
    
    Parameters
    -----------------------------------------
    data : dictionary of clustered features, times, and waveforms, returned
        from load_spikes function
    thresh : detection threshold used for spike sorting
    t_ref : the width of the refractory period, in seconds
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    ignore : a list of the clusters to ignore in the analysis.  Default is cluster
        zero since that is typically the noise cluster.
    
    Returns
    -----------------------------------------
    f_p : dictionary of false positive estimates for each cluster
        if an estimate can't be made, will be NaN.  This means that the 
        estimate is >50%
    f_n : dictionary of false negative estimates for each cluster
    
    '''
    
    f_p_r = refractory(data['times'], t_ref, t_cen, t_exp)
    f_n_t = threshold(data['waveforms'], thresh)
    f_p_o, f_n_o = overlap(data['features'], ignore)
    f_n_c = censored(data['times'], t_cen, t_exp)
    
    # Make a dictionary to store the false positive rates
    f_p = dict.fromkeys(f_p_r.keys())
    
    # Make a dictionary to store the false negative rates
    f_n = dict.fromkeys(f_p_r.keys())
    
    for clst in f_p_r.iterkeys():
        if np.isnan(f_p_r[clst]):
            f_p[clst] = np.nan
        else:
            f_p[clst] = f_p_r[clst] + f_p_o[clst]
        
        if f_n_o[clst] == None:
            f_n[clst] = np.nan
        else:
            f_n[clst] = f_n_t[clst] + f_n_o[clst] + f_n_c[clst]
        
    return f_p, f_n
    
def batch_metrics(unit_list, threshold, t_ref, t_cen):
    ''' This here function runs metrics on a batch of data.  Pass in units from the catalog.
    '''
    
    from scipy import unique
    
    samp_rate = 30000.
    n_samples = 30
    n_chans = 4
    
    # Find common Sessions
    sessions = unique([unit.session for unit in unit_list])
    
    for session in sessions:
        
        units = session.units
        tetrodes = unique([unit.tetrode for unit in units])
        
        for tetrode in tetrodes:
            data = load_spikes(session.path, tetrode,  samp_rate, n_samples, n_chans)
            f_p, f_n = metrics(data, threshold, t_ref, t_cen, session.duration)
            # Doing this because sometimes there is no cluster 0 sometimes
            f_p.setdefault(1)
            f_n.setdefault(1)
            units = [ unit for unit in session.units if unit.tetrode == tetrode] 
            for unit in units:
                unit.falsePositive = f_p[unit.cluster]
                unit.falseNegative = f_n[unit.cluster]
    