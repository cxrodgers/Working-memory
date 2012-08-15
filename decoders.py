""" This module will contain decoders and other similar analyses.
Author:  Mat Leonard
Last modified:  7/02/2012
"""

from myutils import find
import numpy as np

class Bayesian(object):
    
    def __init__(self, goals, rates):
        
        self._goals = goals
        
        if len(np.shape(rates)) < 3:
            # This means we're working with a single unit
            self._Nunits = 1
            self._Ntrials, self._dims = np.shape(rates)
            self._rates = [rates]
            
        else:
            self._Nunits, self._Ntrials, self._dims = np.shape(rates)
            self._rates = rates
        
        
        self._Ngoals = len(np.unique(goals))
        self._goalnames = np.unique(goals)
        
        # Store the trial indices for each goal
        self._gdict = dict.fromkeys(np.unique(goals))
        self._means = self._gdict.copy()
        self._covs = self._gdict.copy()
        
        self.priors = [ self._gdict.copy() for ii in np.arange(self._Ntrials) ]
        self.p_gr = [ self._gdict.copy() for ii in np.arange(self._Ntrials) ]
        
        # Need to identify the trials for each goal choice
        for goal in self._gdict.iterkeys():
            self._gdict[goal] = find(goals == goal)
            
        # Calculate means and covariances for the prior
        for goal, ind in self._gdict.iteritems():
            self._means[goal] = np.array([ np.mean(unit[ind], axis = 0) 
                for unit in self._rates])
            if self._dims < 2:
                self._covs[goal] = np.array([ np.std(unit[ind]) 
                for unit in self._rates])
            else:
                self._covs[goal] = np.array([ np.cov(unit[ind], rowvar = 0) 
                    for unit in self._rates])
        
        
    def decode(self):
        
        self.decoded = np.zeros(self._Ntrials, dtype = self._goalnames.dtype)
        
        # Calculate the priors
        self._priors()
        
        self.p_g = dict.fromkeys(self._gdict.keys())
        for goal in self.p_g.iterkeys():
            self.p_g[goal] = len(self._gdict[goal])/np.float(self._Ntrials)
        
        for tid, trial in enumerate(self.priors):
            for goal in self._goalnames:
                self.p_gr[tid][goal] = trial[goal]*self.p_g[goal]
            
            
            self.decoded[tid] = sorted(self.p_gr[tid].items(), 
                key = lambda tr: tr[1])[-1][0]

        return self.decoded
    
    def performance(self):
        
        perf = 1- np.sum(np.abs(self._goals - self.decoded))/np.float(self._Ntrials)
        return perf
        
    def _priors(self):
        
        p_rg_i = np.zeros(self._Nunits)
        
        # Calculate the prior for one trial at a time
        for trial in np.arange(self._Ntrials):
            # Calculate the prior for each goal
            for goal in self._goalnames:
                # Take the product over the units
                for uid, urates in enumerate(self._rates):
                    mean = self._means[goal][uid]
                    cov = self._covs[goal][uid]
                    if self._dims < 2:
                        p_rg_i[uid] = normal(urates[trial], mean, cov)
                    else:
                        p_rg_i[uid] = multinormal(urates[trial], mean, cov)
                
                self.priors[trial][goal] = np.prod(p_rg_i)  

def spike_rates(trials, spikes, dims):
    
    # Number of bins for the firing rate calculation
    bN = dims
    
    # Initialize array for storing rate vectors
    rate_mat = np.zeros((len(spikes), len(trials), bN))
    
    for ii, unit in enumerate(spikes):
        pg = trials['PG time'][ii]
        c_out = trials['C time'][ii][1]
        
        rate_vecs = np.array([ np.histogram(spks, bins = bN, 
            range = (pg, c_out))[0]/((c_out-pg)/float(bN)) for spks in unit ])
        
        rate_mat[ii] = rate_vecs
    
    return rate_mat

def multinormal(vec, mean, cov):
    
    dims = len(mean)
    dcov = np.linalg.det(cov)
    icov = np.linalg.inv(cov)
    dif = vec - mean
    expo = np.exp(-0.5 * np.dot(np.dot(dif, icov), dif))
    norm = (2* np.sqrt( np.power(np.pi, dims) * dcov))
    
    return (expo / norm)

def normal(rate, mean, sig):
    
    expo = np.exp(-0.5*(rate-mean)**2/sig**2)
    norm = np.sqrt(2*np.pi*sig**2)
    
    return (expo / norm)
    
    
    
    