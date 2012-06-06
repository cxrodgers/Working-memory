"""  A module containing functions that calculate metrics for testing the 
quality of clustered data.

Author:  Mat Leonard
Last modified: 6/5/2012
"""

import numpy as np
import kkpandas
from scipy import optimize as opt

def load_spikes(features_file, clusters_file, samp_rate):
    ''' This method takes the feature and cluster files in KlustaKwik format 
    and pulls out the spike times for each cluster.
    
    Arguments:
    features_file : string path to the features file in KlustaKwik format
    clusters_file : string path to the clusters file in KlustaKwik format
    samp_rate : the sampling rate of the recording
    '''

    # Get spike times
    features = kkpandas.read_fetfile(features_file)
    spike_times = features.spike_time.values
    
    # Get spike cluster labels
    clusters = kkpandas.read_clufile(clusters_file)
    
    # Cluster numbers
    cluster_nums = np.unique(clusters.values)
    
    # I think the KlustaKwik format is annoying...
    
    # Grouping the indices by cluster
    cluster_ind = [ np.nonzero(clusters.values == n)[0] for n in cluster_nums ]
    
    # Get the spikes for each cluster
    cluster_spikes = [ spike_times[ind]/np.float(samp_rate) for ind in cluster_ind ]
    
    # Make a dictionary where each key is the cluster number and the value
    # is an array of the spike times in that cluster
    clusters = dict(zip(cluster_nums, cluster_spikes)) 
    
    # Let's make sure the spike times for each cluster is sorted correctly
    for spikes in clusters.itervalues():
        spikes.sort()
    
    return clusters

def false_pos(clusters, t_ref, t_cen, t_exp):
    ''' Returns an estimation of false positives in a cluster based on the
    number of refractory period violations.  Returns NaN if an estimate can't
    be made, or if the false positive estimate is greater that 0.5.  This is 
    most likely because of too many spikes in the refractory period.
    
    Arguments:
    clusters : dictionary of clusters from the load_spikes function
    t_ref : the width of the refractory period, in seconds
    t_cen : the width of the censored period, in seconds
    t_exp : the total length of the experiment, in seconds
    
    '''
   
    # Make the function we will use to solve for the false positive rate
    func = lambda f: 2 * f * (1 - f) * N**2 * (t_ref - t_cen) / t_exp - ref
    
    # Make a list to store the false positive rates
    f_p_1 = dict.fromkeys(clusters.keys())
    
    # Calculate the false positive rate for each cluster.
    for clst, spikes in clusters.iteritems():
        # This finds the time difference between consecutive peaks
        isi = np.array([ jj - ii for ii, jj in zip(spikes[:-1], spikes[1:]) ])
        
        # Now we need to find the number of refractory period violations
        ref = np.sum(isi <= t_ref)
        
        # The number of spikes in the cluster
        N = len(spikes)
        
        # Then the false positive rate
        try:
            f_p_1[clst] = opt.brentq(func, 0, 0.5)
        except:
            f_p_1[clst] = np.nan
    
    return f_p_1
