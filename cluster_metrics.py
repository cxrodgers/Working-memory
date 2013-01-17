"""  A module containing functions that calculate metrics for testing the 
quality of clustered data.

Load data from KlustaKwik files using the load_spikes() function.
Get false positive and false negative estimates using metrics()

Author:  Mat Leonard
Last modified: 8/2/2012
"""

import numpy as np

from scipy.special import erf

from sklearn import mixture
import sklearn

from itertools import combinations

# Not in this directory, depends on how it's stored
try:
    from KKFileSchema import KKFileSchema
except ImportError:
    from kkpandas.KKFileSchema import KKFileSchema

import os

# This is in this directory
import kkio


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
        #isi = np.array([ jj - ii for ii, jj in zip(times[:-1], times[1:]) ])
        isi = np.diff(times)
        
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

def threshold(clustered_waveforms, thresh, warn_on_thresh_violation=True, peak_sample=6, n_samp=24):
    ''' Returns the rate of false negatives caused by spikes below the 
    detection threshold
    
    Fits a Gaussian to the distribution of waveform peaks. If this Gaussian
    looks truncated at the threshold, then the area beyond this truncation
    is an estimate of the number of missing spikes.
    
    Parameters
    -----------------------------------------
    peak_sample, n_samp : where the waveform peaks "should" be
        If you don't provide this, the peaks over the whole waveform are used.    
    warn_on_thresh_violation : if any peaks are detected that are below
        threshold (which should never happen), issue a warning
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
        if peak_sample is None:
            peaks = np.min(spikes, axis=1)
        else:
            peaks = np.min(spikes[:, peak_sample::n_samp], axis=1)
        
        # check if the threshold make sense with the data
        mp = peaks.max()
        if warn_on_thresh_violation and mp > thresh + .1: # a little slop
            print "warning: got peak at %r but thresh is %r" % (mp, thresh)
        
        # Find the moments of the data distribution
        mu = peaks.mean()
        sigma = peaks.std()
        
        # The false negative percentage is the cumulative Gaussian function
        # up to the detection threshold
        cdf = 0.5*(1 + erf((thresh - mu)/np.sqrt(2*sigma**2)))
        f_n[clst] = (1-cdf) 
    
    return f_n

def fraction_during_refractory(clustered_times, t_ref):
    """Simple estimate of false positives: # spikes in refractory / total.
    
    If the neuron were Poisson, we'd expect to see FR * t_ref spikes
    during each window of length t_ref throughout the experiment, so a total
    of FR * t_ref * t_exp / t_ref spikes, which simplifies to just the total
    number of spikes.
    
    So, this can be interpreted as a lowball estimate of false positives
    (only the ones that happened to occur during refractory period), or as
    the height of the autocorrelation compared to its mean.
    """
    f_p = {}
    for uid, times in clustered_times.items():
        # Count violations
        isi = np.diff(times)
        n_violations = sum(isi < t_ref)
        
        # Divide by total number of spikes
        f_p[uid] = n_violations / float(len(times))
    return f_p


def overlap_with_svm(clustered_features, include_list=None, exclude_list=None):
    """Evaluate cluster overlap by the prediction quality of an SVM.
    
    Fit an SVM to the data (with labels). Take the predictions of the SVM
    on the same dataset. We will use the false positives or false negatives
    of these predictions as estimates of the false positives or false negatives
    in the original data. 
    
    The FPR for cluster I is the number of times the SVM incorrectly 
    predicted that a spike from cluster !I was actually in cluster I, 
    divided  by the total number of spikes it predicted to be in I. 
    This keeps the rate between 0 and 1.
    It can be NaN if the algorithm never predicted anything to be in I.
    
    The FNR for cluster I is the number of times the SVM incorrectly 
    predicted that a spike from cluster I was actually in cluster !I, 
    divided by the number of spikes actually in cluster I.
    This keeps the rate between 0 and 1.
    It should never be NaN, unless you provided empty clusters.
    
    Caveats:
        1)  Will be an underestimate of the true error rates because we test
            on the training set. Should really cross-validate.

        2)  Could be an overestimate of the true error rate because the SVM
            might be doing a worse job than the original clustering algorithm.
            Just because the SVM makes an error doesn't mean that the spike
            was actually incorrectly labeled
    
    Parameters:
    clustered_features : dict
        {cluster_id : feature array of shape (N_spikes, N_features)}
    include_list : only clusters in this list will be included
    exclude_list : exclude clusters in this list
    
    Note that including or excluding clusters will probably (definitely?)
    inflate the scores of the remaining units.
    
    Returns: f_p, f_n, confusion
    f_p, f_n : dicts
        {cluster_id : FPR or FNR for that cluster}
        Will be None for units that in `ignore`
    confusion : 2d array
        Has shape (len(clustered_features), len(clustered_features))
        The entry at row i and col j is the number of times a unit that you
        clustered as i was clustered as j by the SVM.
        The rows and columns are sorted by the keys in clustered_features.
    """
    # Figure out which units to include
    unit_ids = sorted(clustered_features.keys())
    if include_list:
        unit_ids = filter(lambda uid: uid in include_list, unit_ids)
    elif exclude_list:
        unit_ids = filter(lambda uid: uid not in exclude_list, unit_ids)
    
    # Concatenate the data and original labels into the shape expected
    # by the classifier
    data_vecs = np.concatenate([clustered_features[k] for k in unit_ids])
    labels = np.concatenate([
        k * np.ones(len(clustered_features[k]), dtype=np.int)
        for k in unit_ids])

    # Fit the data and get predictions
    lsvc = sklearn.svm.LinearSVC()
    lsvc.fit(data_vecs, labels)
    preds = lsvc.predict(data_vecs)

    # Create a confusion matrix .. true unit on rows, predictions on cols
    confusion = np.empty((len(unit_ids), len(unit_ids)), dtype=np.int)
    for nr, true_unit in enumerate(unit_ids):
        for nc, confused_unit in enumerate(unit_ids):
            confusion[nr, nc] = sum(preds[labels == true_unit] == confused_unit)

    # false positive: probability of predicting I when actually !I
    # so, we sum the rows together, and subtract off the # of hits
    false_positives = confusion.sum(axis=0) - np.diag(confusion)

    # The FPR is the number of false positives in the SVC predictions, 
    # divided by number of times that prediction was made
    # So, at worst, every single prediction of I is incorrect: FPR = 1
    false_positive_rate = false_positives / confusion.sum(axis=0).astype(np.float)

    # false negative: probability of labeling !I when actually I
    # so, we sum the columns together, and subtract off the # of hits
    false_negatives = confusion.sum(axis=1) - np.diag(confusion)

    # The FNR is the number of false negatives in the SVC predictions, 
    # divided by number of data points originally labelled that
    # So, at worst, I was never predicted: FNR = 1
    false_negative_rate = false_negatives / confusion.sum(axis=1).astype(np.float)    
    
    # Convert to dict
    f_p = dict([(uid, fpr) for uid, fpr in zip(unit_ids, false_positive_rate)])
    f_n = dict([(uid, fnr) for uid, fnr in zip(unit_ids, false_negative_rate)])
    for uid in clustered_features.keys():
        if uid not in f_p:
            f_p[uid] = None
        if uid not in f_n:
            f_n[uid] = None
    
    return f_p, f_n, confusion

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
        
        # The data we are going to fit the model to, which is unlabelled
        # concatenation of the two clusters.
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
        
        #~ # This is the probability of the spikes in K belonging to each of
        #~ # the 2 fit clusters.
        #~ gppk = gmm.predict_proba(c_feat[k])
        
        #~ # Assign the "error" as the less probable of the two
        #~ erridx_k = gppk.mean(axis=0)).argmin()
        #~ perr_k = gppk[erridx_k]
        
        #~ # This is the probability of the spikes in I belonging to each of
        #~ # the 2 fit clusters.
        #~ gppi = gmm.predict_proba(c_feat[i])
        
        #~ # Assign the "error" as the less probable of the two
        #~ erridx_i = gppk.mean(axis=0)).argmin()
        #~ perr_i = gppk[erridx_k]

        # This calculates the average probability (over k) that a spike in 
        # cluster k belongs to cluster i - false positives in k
        f_p_k_i = 1/N_k*np.min(np.sum(gmm.predict_proba(c_feat[k]), axis=0))
        #~ f_p_k_i = perr_k.sum() / N_k

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
        # False positives is the sum of the fraction of spikes from this
        # cluster that probably belong to all other clusters
        f_p[clst] = np.sum(f_p[clst])
        
        # False negatives is possibly the max?
        f_n[clst] = np.sum(f_n[clst])#np.max(f_n[clst])
        
    return f_p, f_n

def censored(clustered_times, t_cen, t_exp):
    ''' Returns the estimated false negative rate caused by spikes censored
    after a detected spike
    
    The idea is that a spike cannot be detected if another spike was 
    already detected within `t_cen` seconds. So, if we assume spikes occur
    independently, we can multiply the firing rate of all other neurons
    combined by the censored time, to estimate the chances that another spike
    occurred.
    
    If you prioritized bigger waveforms over smaller, then the true censored
    rate will be higher for the units with smaller waveforms and lower for
    the units with bigger waveforms. But this function does not take that
    into account.
    
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
    