""" This module will contain decoders and other similar analyses.
Author:  Mat Leonard
Last modified:  7/02/2012
"""

from myutils import find
import numpy as np

class Bayesian(object):
    '''A Bayesian decoder for decoding goal choice from spike rates'''
    def __init__(self, goals, rates):
        '''Parameters
        -----------------------------------------
        goals : an array of the experimental data for the goal chosen
        rates : a list or array of the rates for each trial, for each unit.  
            The format should have each item as a unit and each item in a unit 
            is the spike rate for that trial.
            So np.array([ Unit1[[trial1], [trial2], ..., [trialN]], 
                Unit2[[trial1], [trial2], ..., [trialN], ..., UnitN[ ... ] ])
            Trial rates can be a single number or a vector.
        
        '''
        
        if len(np.shape(rates)) < 3:
            # This means we're working with a single unit
            self._Nunits = 1
            self._Ntrials, self._dims = np.shape(rates)
            self._rates = (rates,)
            
        else:
            self._Nunits, self._Ntrials, self._dims = np.shape(rates)
            self._rates = rates
        
        # We're going to split the data set in two.  Train on the first half,
        # test on the second half
        self._Ntrain = self._Ntrials/2
        self._Ntest = self._Ntrials - self._Ntrain
        
        self._traingoals = goals[:self._Ntrain]
        self._testgoals = goals[self._Ntrain:]
        
        self._trainrates = [ unit[:self._Ntrain] for unit in self._rates ]
        self._testrates = [ unit[self._Ntrain:] for unit in self._rates ]
        
        # Get the different goal names, I set this up so the goals can be any
        # data type, integers, strings, whatever.
        self._goalnames = np.unique(goals)
        
        # Make some dictionaries for storing data and information
        self._gdict = dict.fromkeys(self._goalnames)
        self._means = self._gdict.copy()
        self._covs = self._gdict.copy()
        
        self.priors = [ self._gdict.copy() for ii in np.arange(self._Ntest) ]
        self.p_gr = [ self._gdict.copy() for ii in np.arange(self._Ntest) ]
        
        # Need to identify the trials for each goal choice, in the training data
        for goal in self._goalnames:
            self._gdict[goal] = find(self._traingoals == goal)
            
        # Calculate means and covariances for the prior from the training data
        for goal, ind in self._gdict.iteritems():
            self._means[goal] = np.array([ np.mean(unit[ind], axis = 0) 
                for unit in self._trainrates])
            if self._dims < 2:
                self._covs[goal] = np.array([ np.std(unit[ind]) 
                for unit in self._trainrates])
            else:
                self._covs[goal] = np.array([ np.cov(unit[ind], rowvar = 0) 
                    for unit in self._trainrates])
        
    def decode(self):
        ''' Decodes the goal choices for the test data from the spike rates '''
        self.decoded = np.zeros(self._Ntest, dtype = self._goalnames.dtype)
        
        # Calculate the priors
        self._priors()
        
        self.p_g = dict.fromkeys(self._gdict.keys())
        
        # Calculate P(g)
        for goal in self._goalnames:
            self.p_g[goal] = len(self._gdict[goal])/np.float(self._Ntrain)
        
        # Calculate P(g|r) = P(r|g) * P(g) (proportional)
        for trial, prior in enumerate(self.priors):
            for goal in self._goalnames:
                self.p_gr[trial][goal] = prior[goal]*self.p_g[goal]
            
            # Find the max posterior goal
            self.decoded[trial] = sorted(self.p_gr[trial].items(), 
                key = lambda tr: tr[1])[-1][0]

        return self.decoded
    
    def performance(self):
        ''' Returns the performance of the decoder, the number of correctly
        decoded trials from the test set divided by the number of trials in the
        test set.'''
        
        perf = np.sum(self._testgoals == self.decoded)/np.float(self._Ntest)
        return perf
        
    def _priors(self):
        ''' This method calculates the prior distributions, P(r|g), for the test
        data using parameters calculated from the training data set. '''
        
        p_rg_i = np.zeros(self._Nunits)
        
        # Calculate the prior for one trial at a time, in the test set
        for trial in np.arange(self._Ntest):
            # Calculate the prior for each goal
            for goal in self._goalnames:
                # Take the product over the units
                for uid, urates in enumerate(self._testrates):
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
    
    
    
    