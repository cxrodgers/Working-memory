""" Okay, this one is going to take the behavior and neural data then generate
timelocked raster plots and peths (peri-event time histograms).  Maybe I
should actually call them Pre Response Time Histograms.  I'll figure it out.
"""

import os
import re
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io
from scipy.stats.mstats import zscore
from itertools import combinations
from numpy.linalg import norm
import myutils as ut

def timelock(datadir, tetrode, time_zero = 'RS', n_shift = 0, scaled = False):
    """ This function takes the behavior and spiking data, then returns
    the spikes timelocked to the 'Response,' when the stimulus plays.
    
    Arguments:
    datadir : path to the directory where the data files are stored.
        The necessary data files are behavior, clusters, syncing, and onsets from
        the recording data.
    
    time_zero : The event that you timelock to, where you set t = 0.
        'RS' sets it to the response, defined as the time when the stimulus plays
        'PG' sets it to the previous goal, defined as the time when the rat
            poked his nose in the previous port
        'FG' sets it to the future goal, defined as the time when the rat pokes
            his nose in the next port
    n_shift : Sometimes the data will be split into multiple files.  If this happens,
    this parameter will shift everything forward trial to trial number n_shift.
    
    scaled : True if you want the spikes to be scaled so each trial is the same length
    
    A typical use would be "trials, spikes = timelock('/path/to/data')"
    """
    

    filelist = os.listdir(datadir)
    filelist.sort()
    
    reg = [ re.search('(\w+)_(\w+).(\w+)', filename) for filename in filelist ]
    reg = [ r for r in reg if r != None ]
    
    ext = ['bhv', 'cls', 'syn', 'ons']
    data = dict.fromkeys(ext)
    
    for r in reg:
        if r.group(3) in ext:
            if r.group(3) == 'cls':
                file = '%s%s.%s' % (datadir, r.group(0), tetrode)
                name = r.group(1)
                date = r.group(2)
            else:
                file = '%s%s' % (datadir, r.group(0))
            with open(file,'r') as f:
                data[r.group(3)] = pkl.load(f)
    
    # Checking to make sure the data files were loaded
    if None in data.viewvalues():
        for key, value in data.iteritems():
            if value == None:
                raise Exception, '%s file wasn\'t loaded properly' % key
                break

    
    # Get the stimulus onset for each trial
    b_onsets = data['bhv']['onsets'];
    consts = data['bhv']['CONSTS'];
    
    # Create a structured array to keep all the relevant trial information
    records = [('2PG port', 'i8'), ('PG port','i8'), ('FG port','i8'), \
        ('2PG outcome', 'i8'), ('PG outcome','i8'), ('FG outcome','i8'), \
        ('PG time','f8'), ('RS time', 'f8'), ('FG time', 'f8'), \
        ('PG response','i8'), ('Response','i8'), ('Trial length', 'f8'), \
        ('Block', 'i8'), ('Scale', 'f8', 2), ('Leave PG', 'f8'), \
        ('C time', 'f8', 2)];

    trials = np.zeros((len(b_onsets),),dtype = records)

    # Populate the array
    
    bdata = data['bhv']['TRIALS_INFO']
    
    # Correct port for the second previous goal
    trials['2PG port'] = np.concatenate((np.zeros(2), \
        bdata['CORRECT_SIDE'][:len(b_onsets)-2]));
    
    # Correct port for the previous goal
    trials['PG port'] = np.concatenate((np.array((0,)), \
        bdata['CORRECT_SIDE'][:len(b_onsets)-1]));

    # Correct port for the future goal
    trials['FG port'] = bdata['CORRECT_SIDE'][:len(b_onsets)];

    # Was the previous response a hit or an error
    trials['2PG outcome'] = np.concatenate((np.zeros(2), \
        bdata['OUTCOME'][:len(b_onsets)-2]))
    
    # Was the previous response a hit or an error
    trials['PG outcome'] = np.concatenate((np.array((0,)), \
        bdata['OUTCOME'][:(len(b_onsets)-1)]))

    # Was the future response a hit or an error
    trials['FG outcome'] = bdata['OUTCOME'][:len(b_onsets)]
    
    # Was the trial cued or uncued
    trials['Block'] = bdata['BLOCK'][:len(b_onsets)]
    
    # What direction did the rat go (left = 1, right = 2)
    for ii in np.arange(len(b_onsets)):
        # If the trial was a hit, then the rat went to the correct port for that trial
        if trials['FG outcome'][ii] == consts['HIT']:
            trials['Response'][ii] = trials['FG port'][ii]
        # If it was an error, then went to the other port
        elif trials['FG outcome'][ii] == consts['ERROR']:
            if trials['FG port'][ii] == consts['LEFT']:
                trials['Response'][ii] = consts['RIGHT']
            elif trials['FG port'][ii] == consts['RIGHT']:
                trials['Response'][ii] = consts['LEFT']
    
    # PG response is just FG response shifted by one trial
    trials['PG response'] = np.concatenate((np.array((0,)),trials['Response'][:-1]));
    
    # Initialize the scaling records, will be ones if not scaled at all
    trials['Scale'] = np.ones((len(trials),2));
    
    # Now we'll get the event times
    for ii in np.arange(len(b_onsets)):
        
        times = data['bhv']['peh'][ii];
        
        # Get previous goal times
        if ii > 0:
            prev_times = data['bhv']['peh'][ii-1];
            
            if trials['PG outcome'][ii] == consts['HIT']:
                pg_time = prev_times['states']['hit_istate'].min();
            elif trials['PG outcome'][ii] == consts['ERROR']:
                pg_time = prev_times['states']['error_istate'].min();
            elif trials['PG outcome'][ii] == consts['CHOICE_TIME_UP']:
                pg_time = prev_times['states']['choice_time_up_istate'].min();
        else:
            pg_time = b_onsets[ii];
        
        # Now get the future goal times
        if trials['FG outcome'][ii] == consts['HIT']:
            fg_time = times['states']['hit_istate'].min();
        elif trials['FG outcome'][ii] == consts['ERROR']:
            fg_time = times['states']['error_istate'].min();
        elif trials['FG outcome'][ii] == consts['CHOICE_TIME_UP']:
            fg_time = times['states']['choice_time_up_istate'].min();
        
        if time_zero == 'RS':
            # Time lock to the response, which I'm saying is when the 
            # stimulus plays
            trials['PG time'][ii] = pg_time - b_onsets[ii];
            trials['RS time'][ii] = 0
            trials['FG time'][ii] = fg_time - b_onsets[ii];
            
        elif time_zero == 'PG':
            # Time lock to the previous goal, which I have set to the time 
            # when the rat pokes in the port.  Might change this to when 
            # the rat leaves the port.
            trials['PG time'][ii] = 0
            trials['RS time'][ii] = b_onsets[ii] - pg_time
            trials['FG time'][ii] = fg_time - pg_time
    
        elif time_zero == 'FG':
            # Time lock to the future goal, when the rat gets the future goal
            trials['PG time'][ii] = pg_time - fg_time
            trials['RS time'][ii] = b_onsets[ii] - fg_time
            trials['FG time'][ii] = 0
        
        trials['Trial length'][ii] = fg_time - pg_time;
        
        # Get the time when the rat leaves the PG port
        if (trials['PG response'][ii] == consts['RIGHT']) &(ii != 0):
            
            rpokes = prev_times['pokes']['R']
            
            pg_leave = np.nanmax( rpokes[rpokes < b_onsets[ii] ])
        
        elif (trials['PG response'] [ii]== consts['LEFT']) & (ii != 0):
            
            lpokes = prev_times['pokes']['L']
            
            pg_leave = np.nanmax( lpokes[ lpokes < b_onsets[ii] ])
            
        if ii == 0:
            pg_leave = pg_time
        
        # Have to set the time based on the event I'm timelocking to
        trials['Leave PG'][ii] = trials['PG time'][ii] + (pg_leave - pg_time)
    
        # Now get the center port times
        trials['C time'][ii][0] = trials['RS time'][ii] - (b_onsets[ii] - 
            np.nanmin(times['states']['hold_center']))
        try: 
            trials['C time'][ii][1] = trials['RS time'][ii] + \
                (np.nanmax(times['states']['hold_center2']) - b_onsets[ii])
        except:
            print times['states']['hold_center']
            print times['states']['hold_center2']
            trials['C time'][ii][1] = trials['C time'][ii][0]

    # Get the scaling factors
    if scaled:
        if time_zero == 'RS':
            for jj in np.arange(len(trials)):
                scale_neg = -7./trials['PG time'][jj];
                scale_pos = 1./trials['FG time'][jj];
                trials['Scale'][jj] = np.array((scale_neg, scale_pos));
        if time_zero == 'PG':
            for jj in np.arange(len(trials)):
                scale_neg = 7./trials['RS time'][jj];
                scale_pos = 8./trials['FG time'][jj];
                trials['Scale'][jj] = np.array((scale_neg, scale_pos));
        if time_zero == 'FG':
            for jj in np.arange(len(trials)):
                scale_neg = -8./trials['PG time'][jj];
                scale_pos = -1./trials['RS time'][jj];
                trials['Scale'][jj] = np.array((scale_neg, scale_pos));
    
    
    # Okay, now get the spikes and time lock to the response, which we are
    # defining as the center poke, the stimulus onset.

    # Let's first sync up the behavior trials and recording onsets

    # If there were more than one neural data files, then you might have
    # to shift things some.  Set this to the number of behavior trials 
    # skipped before the recording data starts.
    #n_shift = 24;

    sync = data['syn'].map_n_to_b

    n_onsets = data['ons']/30000.

    trials_spikes = []

    # Loop over each cluster in the ePhys data
    for cl in data['cls']:
        spikes = [0]*len(b_onsets);
        
        p_times = cl['peaks'];
        
        #~ if n_shift != None:
            #~ shift = 1
        #~ else:
            #~ shift = 0
        
        # Go through each trial and grab the spikes for that trial
        # Need to subtract the first number in sync so everything lines up
        for ii in sync - sync[0]:
            
            if time_zero == 'RS':
                
                # Grab spikes that are between the previous goal time - 30 sec
                # and this trial's onset time
                low_end = p_times > (n_onsets[ii] +
                    trials['PG time'][ii+n_shift] - 30.)
                    
                # Grab spikes that are between the future goal time - 30 sec
                # and this trial's onset time
                high_end = p_times < (n_onsets[ii] +
                    trials['FG time'][ii+n_shift] + 30.)
                
                ind = low_end & high_end
                    
                spikes[ii+n_shift] = p_times[ind]-n_onsets[ii];
                spikes[ii+n_shift].sort();
            
            elif time_zero == 'PG':
                
                # Grab spikes that are between the previous goal time - 30 sec
                # and this trial's onset time
                low_end = p_times > (n_onsets[ii] -
                    trials['RS time'][ii+n_shift] - 30.)
                    
                # Grab spikes that are between the future goal time - 30 sec
                # and this trial's onset time
                high_end = p_times < (n_onsets[ii] + trials['FG time'][ii+n_shift] -
                    trials['RS time'][ii+n_shift] + 30.)
                
                ind = low_end & high_end
                    
                spikes[ii+n_shift] = (p_times[ind] - n_onsets[ii] + 
                    trials['RS time'][ii+n_shift])
                spikes[ii+n_shift].sort();
            
            elif time_zero == 'FG':
                # Grab spikes that are between the previous goal time - 30 sec
                # and this trial's onset time
                low_end = p_times > (n_onsets[ii] -
                    trials['PG time'][ii+n_shift] - 30.)
                    
                # Grab spikes that are between the future goal time - 30 sec
                # and this trial's onset time
                high_end = p_times < (n_onsets[ii] - 
                    trials['RS time'][ii+n_shift] + 30.)
                
                ind = low_end & high_end
                    
                spikes[ii+n_shift] = (p_times[ind] - n_onsets[ii] + 
                    trials['RS time'][ii+n_shift])
                spikes[ii+n_shift].sort()
       
        # Now we're going to scale the spike times
        if scaled:
            for jj in np.arange(len(trials)):
                try: 
                    spikes[jj][spikes[jj]<0] = trials['Scale'][jj][0] * spikes[jj][spikes[jj]<0];
                except:
                    spikes[jj] = spikes[jj];
                try:
                    spikes[jj][spikes[jj]>0] = trials['Scale'][jj][1] * spikes[jj][spikes[jj]>0];
                except:
                    spikes[jj] = spikes[jj];
            
        trials_spikes.append(spikes[n_shift:])
    
    # Now scale event times
    if scaled:
        for jj in np.arange(len(trials)):
            
            if time_zero == 'RS':
                if jj == 0:
                    trials['PG time'][jj] = 0;
                else:
                    trials['PG time'][jj] = trials['Scale'][jj][0] * trials['PG time'][jj];
               
                trials['FG time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
                
                
            
            if time_zero == 'PG':
                if jj == 0:
                    trials['RS time'][jj] = 0;
                else:
                    trials['RS time'][jj] = trials['Scale'][jj][0] * trials['RS time'][jj];
               
                trials['FG time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
            
            if time_zero == 'FG':
                if jj == 0:
                    trials['PG time'][jj] = 0;
                else:
                    trials['PG time'][jj] = trials['Scale'][jj][0] * trials['PG time'][jj];
               
                trials['RS time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
     
            trials['Leave PG'][jj] = trials['Scale'][jj][0] * trials['Leave PG'][jj]
            trials['C time'][jj][0] = trials['Scale'][jj][0] * trials['C time'][jj][0]
            trials['C time'][jj][1] = trials['Scale'][jj][1] * trials['C time'][jj][1]
    
    # Exclude trials longer than 20 seconds
    include = np.nonzero(trials['Trial length'] < 20.)[0]
    trials = trials[include]
    
    for x,spikes in enumerate(trials_spikes):
        trials_spikes[x] = tuple([ spikes[ii] for ii in include ])
       
    return trials[n_shift:], trials_spikes
    
def raster(trials, trial_spikes, range = (-20,2)):
    ''' This function creates a raster plot.  Each row is a trial, each spike
    is shown as a vertical dash.  
   
    Arguments :
    
    trials : trial information, returned from the timelock function
    trial_spikes : spike time information, returned from the timelock function
    '''
    
    # Each row is a different trial, each spikes should be marked 
    # as a vertical dash.
    ind = 1
    
    sorting = np.zeros(len(trials), dtype=[('indices', 'i8'), ('PG time','f8')])
    
    sorting['indices'] = np.arange(0,len(trials))
    sorting['PG time'] = trials['PG time']
    sorting.sort(order='PG time')
    sorting = sorting[::-1] 
    
    for ii in sorting['indices']:
        
        spikes = trial_spikes[ii]
        
        # Checking if that trial has spikes
        try:
            spike_test = spikes.any()
        except:
            spike_test = False
        
        # If it doesn't, just plot a red dash at t = 0
        if spike_test:
            plt.plot(spikes, ind*np.ones((len(spikes),)),'|',color='k');
        else:
            plt.plot(0, ind,'|',color='r');
            
        # Plot markers to denote the PG or RS time
        if trials['PG time'][-1] != 0:
            plt.plot(trials['PG time'][ii],ind,'.r');
        elif trials['PG time'][-1] == 0:
            plt.plot(trials['RS time'][ii],ind,'.r');
            
       
       # Plot markers to denote the RS or FG time
        if trials['FG time'][-1] != 0:
            plt.plot(trials['FG time'][ii],ind,'.b');
        elif trials['FG time'][-1] == 0:
            plt.plot(trials['RS time'][ii],ind,'.b');
        ind = ind + 1;
     
    # Plot a gray line at t = 0 for reference
    plt.plot([0,0],[0,ind],'grey')
    plt.xlim(range)
    plt.show()
    
    return None
    
def zscores(trials, data, bin_width = .2,range = (-10,3), label = None):
    ''' Returns a plot showing the calcuated zscores over multiple trials'''
    
    x, avg_peth, peths = peth(trials, data, bin_width, range, label, to_plot = 0)
    
    zsc = zscore(peths)
    
    extent = x[0], x[-1], 0, np.shape(zsc)[0]
    
    plt.imshow(zsc, interpolation = 'nearest', origin = 'lower', 
        aspect = 'auto', extent = extent)

def peth(trials, data, bin_width = 0.2,range = (-10,3), label = None,
    to_plot = 1):
    ''' Returns a Peri-Event Time Histogram.  Basically, it gives you a rate
    histogram over time.
    '''
    
    n_trials = len(data);
    
    bins = np.diff(range)[0]/bin_width;
    
    # This is here because the very first trial doesn't have a PG, so
    # it makes it weird.  So, skip that trial.
    if trials['Scale'][0,0] == -np.inf:
        skip = 1
    else:
        skip = 0
    
    all_trials = [0]*(n_trials-skip)
    
    for ii in np.arange(len(data))[skip:]:
        histo = np.histogram(data[ii],bins,range);
        x = histo[1][:-1] + bin_width/2.
        one_trial_neg = histo[0][x<=0]/bin_width*trials['Scale'][ii][0]
        one_trial_pos = histo[0][x>0]/bin_width*trials['Scale'][ii][1] 
        all_trials[ii-1] = np.concatenate((one_trial_neg,one_trial_pos))
    
    peth = np.mean(np.array(all_trials[skip:]),axis=0);
   
    if to_plot:   
        plt.plot(x,peth, label = label);
    
    plt.show();
    
    return x, peth, np.array(all_trials)

def avg_rate(data, to_plot = False):
    """ Calculates and returns the average spike rate for each trial."""
    
    trial_spikes = data
    
    # For each trial, divide into 10 windows, and calculate the spike rate,
    # then average the spike rate.
    avg = []
    for spikes in trial_spikes:
        first = spikes[0]
        last = spikes[-1]
        
        bin_width = np.diff((first-0.1, last+0.1))/10.;
        
        try:
            hist = np.histogram(spikes, bins = 10, range = (first-0.1, last+0.1))
            avg.append(np.mean(hist[0])/bin_width)
        except:
            pass
            
    avg = np.array(avg).flatten();
    
    if to_plot:
        plt.figure();
        plt.hist(avg, bins = 40, range = (0,20))
        plt.show()
        
    return avg

def constants():
    
    consts = {'SHORT_CPOKE': 5, 'LEFT': 1, 'CURRENT_TRIAL': 6, 'HIT': 2, 'NONRANDOM': 2, \
          'TWOAC': 3, 'RANDOM': 1, 'NONCONGRUENT': -1, 'FUTURE_TRIAL': 4, 'CONGRUENT': 1,\
          'RIGHT': 2, 'INCONGRUENT': -1, 'ERROR': 1, 'UNKNOWN_OUTCOME': 4, 'WRONG_PORT': 7,\
          'NOT-GO-NOGO': 3, 'SHORTPOKE': 5, 'GO': 1, 'CHOICE_TIME_UP': 3, 'NOGO': 2};
    
    return consts

def get_figs(trials, trial_spikes, bin_width = 0.2, range = (-7,1)):
    
    consts = constants();

    lefts = trials['Response'] == consts['LEFT'];
    rights = trials['Response'] == consts['RIGHT'];
    hits = trials['FG outcome'] == consts['HIT'];
    errs = trials['FG outcome'] == consts['ERROR'];
    pg_hits = trials['PG outcome'] == consts['HIT'];
    pg_errs = trials['PG outcome'] == consts['ERROR'];
    pg_lefts = trials['PG response'] == consts['LEFT'];
    pg_rights = trials['PG response'] == consts['RIGHT'];
    
    pg_left_hits = pg_lefts & pg_hits;
    pg_left_errs = pg_lefts & pg_errs;
    pg_right_hits = pg_rights & pg_hits;
    pg_right_errs = pg_rights & pg_errs;
    
    spk_lefts = [trial_spikes[ii] for ii in lefts.nonzero()[0]];
    spk_rights = [trial_spikes[ii] for ii in rights.nonzero()[0]];
    spk_hits = [trial_spikes[ii] for ii in hits.nonzero()[0]];
    spk_errs = [trial_spikes[ii] for ii in errs.nonzero()[0]];
    spk_pg_hits = [trial_spikes[ii] for ii in pg_hits.nonzero()[0]];
    spk_pg_errs = [trial_spikes[ii] for ii in pg_errs.nonzero()[0]];
    spk_pg_lefts = [trial_spikes[ii] for ii in pg_lefts.nonzero()[0]];
    spk_pg_rights = [trial_spikes[ii] for ii in pg_rights.nonzero()[0]];
        
    spk_pg_left_hits = [trial_spikes[ii] for ii in pg_left_hits.nonzero()[0]];
    spk_pg_left_errs = [trial_spikes[ii] for ii in pg_left_errs.nonzero()[0]];
    spk_pg_right_hits = [trial_spikes[ii] for ii in pg_right_hits.nonzero()[0]];
    spk_pg_right_errs = [trial_spikes[ii] for ii in pg_right_errs.nonzero()[0]];
        
    fig1 = plt.figure()
    peth(trials[lefts],spk_lefts, bin_width, range, label = 'left' + " n = " + str(len(spk_lefts)));
    peth(trials[rights],spk_rights, bin_width, range, label = 'right'+ " n = " + str(len(spk_rights)));
    ax = fig1.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,plt.ylim()[1] + 1],'-',color = 'grey', lw=2)
    #plt.ylim((0,ymax));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig2 = plt.figure()
    peth(trials[hits],spk_hits, bin_width, range, label = 'hits' + ' n = ' + str(len(spk_hits)));
    peth(trials[errs],spk_errs, bin_width, range, label = 'errors' + ' n = ' + str(len(spk_errs)));
    ax = fig2.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,plt.ylim()[1] + 1],'-',color = 'grey', lw=2)
    #plt.ylim((0,ymax));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig3 = plt.figure()
    peth(trials[pg_hits],spk_pg_hits, bin_width, range, label = 'PG hits' + " n = " + str(len(spk_pg_hits)));
    peth(trials[pg_errs],spk_pg_errs, bin_width, range, label = 'PG errors' + " n = " + str(len(spk_pg_errs)));
    ax = fig3.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,plt.ylim()[1] + 1],'-',color = 'grey', lw=2)
   # plt.ylim((0,ymax));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig4 = plt.figure()
    peth(trials[pg_left_hits],spk_pg_left_hits, bin_width, range, 
        label = 'PG left-hit' + " n = " + str(len(spk_pg_left_hits)));
    peth(trials[pg_left_errs],spk_pg_left_errs, bin_width, range, 
        label = 'PG left-error' + " n = " + str(len(spk_pg_left_errs)));
    peth(trials[pg_right_hits],spk_pg_right_hits, bin_width, range, 
        label = 'PG right-hit' + " n = " + str(len(spk_pg_right_hits)));
    peth(trials[pg_right_errs],spk_pg_right_errs, bin_width, range, 
        label = 'PG right-error' + " n = " + str(len(spk_pg_right_errs)));
    ax = fig4.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,plt.ylim()[1] + 1],'-',color = 'grey', lw=2)
    #plt.ylim((0,ymax));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig5 = plt.figure()
    peth(trials,trial_spikes, bin_width, range, label = 'All trials' + " n = " + str(len(trial_spikes)));
    ax = fig5.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,plt.ylim()[1] + 1],'-',color = 'grey', lw=2)
    #plt.ylim((0,ymax));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig6 = plt.figure()
    peth(trials[pg_lefts],spk_pg_lefts, bin_width, range, label = 'PG left' + " n = " + str(len(spk_pg_lefts)));
    peth(trials[pg_rights],spk_pg_rights, bin_width, range, label = 'PG right'+ " n = " + str(len(spk_pg_rights)));
    ax = fig6.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,plt.ylim()[1] + 1],'-',color = 'grey', lw=2)
    #plt.ylim((0,ymax));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    plt.show()
    
def _generate_LLRR(data, bin_width, range, fig_title = None):
    
    ylims=[0]*4
    titles = ['right->left', 'left->left', 'left->right', 'right->right']
    
    subplots = np.arange(221,225)
    
    # Making the peths
    plt.figure()
    for ii, num in enumerate(subplots):
        plt.subplot(num)
        peth(data[ii][0], data[ii][1], bin_width, range)
        plt.title(titles[ii])
        plt.xlim(range)
        ylims[ii] = plt.ylim()[1]
    
    # Scaling all subplots to the same max y
    ymax = np.max(ylims)
    
    for num in subplots:
        plt.subplot(num)
        plt.ylim(0,ymax)
    
    # Now making the raster plots
    plt.figure()
    for ii, num in enumerate(subplots):
        plt.subplot(num)
        raster(data[ii][0], data[ii][1], range)
        plt.title(titles[ii])
    

def LLRR_figs(trials, spikes, bin_width = 0.2, range = (-7,1)):
    """ Makes raster and histogram plots for each trajectory, cued and 
    uncued."""
    
    # Find the trials for all four PG-FG combinations, in the uncued blocks,
    # and FG hits only.
    
    uncued = [0]*4
    uncued[0] = by_condition(trials, spikes,'right', 'left', 'uncued', 'hit')
    uncued[1] = by_condition(trials, spikes,'left', 'left', 'uncued', 'hit')
    uncued[2] = by_condition(trials, spikes,'left', 'right', 'uncued', 'hit')
    uncued[3] = by_condition(trials, spikes,'right', 'right', 'uncued', 'hit')
    
    _generate_LLRR(uncued, bin_width, range)
    
    cued = [0]*4
    cued[0] = by_condition(trials, spikes,'right', 'left', 'cued', 'hit')
    cued[1] = by_condition(trials, spikes,'left', 'left', 'cued', 'hit')
    cued[2] = by_condition(trials, spikes,'left', 'right', 'cued', 'hit')
    cued[3] = by_condition(trials, spikes,'right', 'right', 'cued', 'hit')
    
    _generate_LLRR(cued, bin_width, range)
    
  
def cue_uncue(trials, spikes, bin_width = 0.2, range = (-7,1)):
    ''' This methods makes plots comparing cued and uncued PETHs '''
    
    conditions = [0]*8
    c_spikes = [0]*8
    conditions[0], c_spikes[0] = by_condition(trials, spikes,'right', 'left', 'cued', 'hit')
    conditions[1], c_spikes[1] = by_condition(trials, spikes,'right', 'left', 'uncued', 'hit')
    conditions[2], c_spikes[2] = by_condition(trials, spikes,'left', 'left', 'cued', 'hit')
    conditions[3], c_spikes[3] = by_condition(trials, spikes,'left', 'left', 'uncued', 'hit')
    conditions[4], c_spikes[4] = by_condition(trials, spikes,'left', 'right', 'cued', 'hit')
    conditions[5], c_spikes[5] = by_condition(trials, spikes,'left', 'right', 'uncued', 'hit')
    conditions[6], c_spikes[6] = by_condition(trials, spikes,'right', 'right', 'cued', 'hit')
    conditions[7], c_spikes[7] = by_condition(trials, spikes,'right', 'right', 'uncued', 'hit')
    
    
    # Now plot all four on one figure
    plt.figure()
    
    ylims = [0]*4
    labels = ['Right->left', 'Left->left', 'Left->right', 'Right->right']

    for ii in np.arange(4): 
        plt.subplot(220 + ii + 1)
        
        # Plot averaged PETHs
        x, unavg, uncued = peth(conditions[2*ii+1], c_spikes[2*ii+1],
            bin_width, range, label = 'Uncued')
        x, cuavg, cued = peth(conditions[2*ii], c_spikes[2*ii],
            bin_width, range, label = 'Cued')
        plt.title(labels[ii])
        plt.xlim(range)
        plt.xticks(size = 'large')
        plt.yticks(size = 'large')
        ylims[ii]=plt.ylim()[1]
        
        dx = np.diff(x)[0]
        
        # Now plot standard errors
        stderr_un = np.std(uncued, axis = 0)/np.sqrt(len(uncued))
        stderr_cu = np.std(cued, axis = 0)/np.sqrt(len(cued))
        
        y1 = unavg - stderr_un
        y2 = unavg + stderr_un
        plt.fill_between(x, y1 , y2, where=y2>=y1, facecolor='b', alpha=0.25)
        y1 = cuavg - stderr_cu
        y2 = cuavg + stderr_cu
        plt.fill_between(x, y1, y2, where=y2>=y1, facecolor='g', alpha=0.25)
        
        # Now we need to check for significance
        # Plot a light gray rectangle where signficant
        # sigs = np.zeros(uncued.shape[1], dtype = 'bool')
#         for ii in np.arange(uncued.shape[1]):
#             sig = ut.ranksum_small(uncued[:,ii], cued[:,ii])
#             sigs[ii] = sig[1] <= 0.05
#             if sig[1] <= 0.05:
#                 plt.fill_between([x[ii] - dx/2, x[ii]+dx/2], 0, 20, where = [True, True], 
#                     facecolor = 'k', edgecolor = 'none', alpha = 0.2, dashes = 'dashdot')
    
    plt.legend()
    ymax = np.max(ylims)
    
    plt.subplot(221)
    plt.ylim(0,ymax)
    
    plt.subplot(222)
    plt.ylim(0,ymax)
    
    plt.subplot(223)
    plt.ylim(0,ymax)
    
    plt.subplot(224)
    plt.ylim(0,ymax)
    

def epoch_histogram(trials, spikes, low_event, high_event):
    
    ep, duration = epoch(trials, spikes, low_event, high_event)
    
    counts = np.array([ len(trl) for trl in ep ])
    
    rate = counts/duration
    hbins = 500
    hrange = (0,100)
    
    events, x = np.histogram(rate, bins = hbins, range = hrange)
    
    plt.plot(x[:-1]+hrange[1]/float(hbins), events)
    plt.show()
    
    
def epoch_scatter(list_trials, list_spikes, compare, measure_func = np.mean, label = 'Avg rate'):
    ''' compare: valid options are 'PG', 'FG', 'block', 'repeats'
    '''
    events = [('PG in', 'PG out'), ('PG out', 'Center in'), ('Center in', 'Center out'),
        ('Center out', 'FG in')]
    
    comparisons = ['PG', 'FG', 'block', 'repeats']
    if compare in comparisons:
        pass
    else:
        raise ValueError, '%s not a valid option for compare' % compare
    
    if type(list_spikes) != list:
        list_spikes = [list_spikes]
    if type(list_trials) != list:
        list_trials = [list_trials]
        
    units = []
    
    for trials, spikes in zip(list_trials, list_spikes):
        #1/0
        if compare == 'PG':
            x_trls, x_spks = by_condition(trials, spikes, PG = 'left')
            y_trls, y_spks = by_condition(trials, spikes, PG = 'right')
            x_label = label + ', PG left'
            y_label = label + ', PG right'
        elif compare == 'FG':
            x_trls, x_spks = by_condition(trials, spikes, FG = 'left')
            y_trls, y_spks = by_condition(trials, spikes, FG = 'right')
            x_label = label + ', FG left'
            y_label = label + ', FG right'
        elif compare == 'block':
            x_trls, x_spks = by_condition(trials, spikes, block = 'cued')
            y_trls, y_spks = by_condition(trials, spikes, block = 'uncued')
            x_label = label + ', cued'
            y_label = label + ', uncued'
            
        x_epochs = [epoch(x_trls, x_spks, ev[0], ev[1]) for ev in events]
        y_epochs = [epoch(y_trls, y_spks, ev[0], ev[1]) for ev in events]
        
        x_mean = []
        y_mean = []
        
        for ep in x_epochs:
            
            rates = np.array([ len(spks)/ep[1][ii]  for ii, spks in enumerate(ep[0]) ])
            x_mean.append(_bootstrap(rates, measure_func))
        
        for ep in y_epochs:
            
            rates = np.array([ len(spks)/ep[1][ii]  for ii, spks in enumerate(ep[0]) ])
            y_mean.append(_bootstrap(rates, measure_func))
        
        units.append((x_mean, y_mean))
        
    titles = ['In PG', 'PG to center', 'In center', 'Center to FG']
    
    for ii in range(len(events)):
        plt.figure()
        for unit in units:
            x_mean = unit[0]
            y_mean = unit[1]
            x, xconfs = x_mean[ii]
            y, yconfs = y_mean[ii]
            yerrlow = np.array([y - yconfs[0]])
            yerrhigh = np.array([yconfs[1] - y])
            xerrlow = np.array([x - xconfs[0]])
            xerrhigh = np.array([xconfs[1] - x])
            x = np.array([x])
            y = np.array([y])
            #1/0
            plt.errorbar(x, y, yerr = [yerrlow, yerrhigh], xerr = [xerrlow, xerrhigh],
                fmt='o', ecolor = 'grey', mfc = 'grey')
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(titles[ii])
        plt.plot([0,plt.xlim()[1]], [0,plt.ylim()[1]], '--k')
        #plt.xlim(0,plt.xlim()[1])
        #plt.ylim(0,plt.ylim()[1])
        plt.show()
        
    
def _bootstrap(data, measure_func = np.mean):
    boot = ut.bootstrap(data, measure_func)
    mean = np.mean(boot)
    confidence = mlab.prctile(boot, p=(2.5, 97.5))
    
    return mean, confidence
    
def epoch(trials, spikes, low_event, high_event):
    lows = ['PG in', 'PG out', 'Center in', 'Center out', 'Response']
    highs = ['PG out', 'Center in', 'Center out', 'Response', 'FG in']
    
    if low_event in lows:
        pass
    else:
        raise ValueError, '%s not a valid option for low_event' % low_event
        
    if high_event in highs:
        pass
    else:
        raise ValueError, '%s not a valid option for high_event' % high_event
    
    section = []
    duration = []
    
    for ii, spks in enumerate(spikes):
        # Grabbing spikes between events in the trials
        
        if low_event == 'PG in':
            low = trials['PG time'][ii]
        elif low_event == 'PG out':
            low = trials['Leave PG'][ii]
        elif low_event == 'Center in':
            low = trials['C time'][ii][0]
        elif low_event == 'Center out':
            low = trials['C time'][ii][1]
        elif low_event == 'Response':
            low = trials['RS time'][ii]
        
        if high_event == 'PG out':
            high = trials['Leave PG'][ii]
        elif high_event == 'Center in':
            high = trials['C time'][ii][0]
        elif high_event == 'Center out':
            high = trials['C time'][ii][1]
        elif high_event == 'Response':
            high = trials['RS time'][ii]
        elif high_event == 'FG in':
            high = trials['FG time'][ii]
        
        duration.append(high - low)
        try:
            interval = ut.find((spks>=low) & (spks<=high))
            section.append( spks[interval] )
        except ValueError:
            interval = 1
        
    duration = np.array(duration)
    
    return section, duration
        

def by_condition(trials, spikes = None, PG = 'all', FG = 'all', block = 'all',
        outcome = 'all', pg_outcome = 'all', PPG = 'all'):
    '''  This method returns a list of the trial indices that fit the supplied conditions.
    
    Arguments:
    
    PG (previous goal) can be 'left', 'right', or 'all'
    FG (future goal) can be 'left', 'right', or 'all'
    block can be 'cued', 'uncued', or 'all'
    outcome can be 'hits', 'errors', or 'all'
    PPG (previous previous goal) can be 'left', 'right', or 'all'
    
    '''
    
    consts = constants()
    
    # Find the trials for each individual condition
    ppg_lefts = np.nonzero(trials['2PG port'] == consts['LEFT'])[0]
    ppg_rights = np.nonzero(trials['2PG port'] == consts['RIGHT'])[0]
    
    pg_lefts = np.nonzero(trials['PG port'] == consts['LEFT'])[0]
    pg_rights = np.nonzero(trials['PG port'] == consts['RIGHT'])[0]
    
    fg_lefts = np.nonzero(trials['FG port'] == consts['LEFT'])[0]
    fg_rights = np.nonzero(trials['FG port'] == consts['RIGHT'])[0]
    
    cued = np.nonzero(trials['Block'] == 1)[0]
    uncued = np.nonzero(trials['Block'] == 2)[0]
    
    hits =  np.nonzero(trials['FG outcome'] == consts['HIT'])[0]
    errors = np.nonzero(trials['FG outcome'] == consts['ERROR'])[0]
    
    pg_hits =  np.nonzero(trials['PG outcome'] == consts['HIT'])[0]
    pg_errors = np.nonzero(trials['PG outcome'] == consts['ERROR'])[0]
    
    if PG == 'left':
        pg_set = set(pg_lefts)
    elif PG == 'right':
        pg_set = set(pg_rights)
    elif PG == 'all':
        pg_set = set.union(set(pg_lefts),set(pg_rights))
        
    if FG == 'left':
        fg_set = set(fg_lefts)
    elif FG == 'right':
        fg_set = set(fg_rights)
    elif FG == 'all':
        fg_set = set.union(set(fg_lefts),set(fg_rights))

    if PPG == 'left':
        ppg_set = set(ppg_lefts)
    elif PPG == 'right':
        ppg_set = set(ppg_rights)
    elif PPG == 'all':
        ppg_set = set.union(set(ppg_lefts),set(ppg_rights))

    if block == 'cued':
        block_set = set(cued)
    elif block == 'uncued':
        block_set = set(uncued)
    elif block == 'all':
        block_set = set.union(set(cued), set(uncued))
        
    if outcome == 'hit':
        outcome_set = set(hits)
    elif outcome == 'error':
        outcome_set = set(errors)
    elif outcome == 'all':
        outcome_set = set.union(set(hits), set(errors))
        
    if pg_outcome == 'hit':
        pg_outcome_set = set(pg_hits)
    elif pg_outcome == 'error':
        pg_outcome_set = set(pg_errors)
    elif pg_outcome == 'all':
        pg_outcome_set = set.union(set(pg_hits), set(pg_errors))
        
    trial_set = set.intersection(pg_set, fg_set, ppg_set, block_set,
        outcome_set, pg_outcome_set)
    
    trials_ind = np.sort(list(trial_set))
    trials_out = trials[trials_ind]
    
    if spikes != None:
        
        spikes_out = [ spikes[trl] for trl in trials_ind ]
        return trials_out, spikes_out
        
    return trials_out

def reliability(spikes, width):
    ''' This function will measure the reliability of spikes for each
    trajectory and compare between cued and uncued 
    
    Arguments:
    spikes : a list of spike times for each trial that you want to find
        the reliability across
    width : width of the Gaussian used to convolve with the spike train,
        enter in units of seconds    
    '''
    
    # Convolve each spike train with a Gaussian.
    # First step is to make each trial a spike train
    trains = [ np.histogram(trial, bins = 8*1000, range = (-7,1))[0]
        for trial in spikes ]
   
    # Make the Gaussian
    x = np.arange(-1,1,0.001)
    gaus = 1/(np.sqrt(2*np.pi)*(width))*np.exp(-x*x/2*width**2)
    
    # Now convolve each train
    conv = [ np.convolve(train, gaus) for train in trains ]
    
    # Now find correlations for each pair
    corrs = []
    itercomb = combinations(np.arange(len(conv)),2)
    for ii, jj in itercomb:
        corrs.append(np.dot(conv[ii], conv[jj]) / 
            (norm(conv[ii]) * norm(conv[jj])))
    
    # And sum over every thing to get the reliability measure
    R_corr = 2./(len(conv))/(len(conv)-1)*sum(corrs)
            
    return R_corr
