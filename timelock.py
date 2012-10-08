""" Okay, this one is going to take the behavior and neural data then generate
timelocked raster plots and peths (peri-event time histograms).  Maybe I
should actually call them Pre Response Time Histograms.  I'll figure it out.
"""

import os
import re
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats.mstats import zscore
from itertools import combinations
from numpy.linalg import norm
import myutils as ut

def timelock(datadir, time_zero = 'RS', n_shift = 0, scaled = False):
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
    
   # First we need to import the behavior and neural data, also the syncing
#     # information.
# 
#     # Get list of files in datadir
#     filelist = os.listdir(datadir);
#     filelist.sort();
# 
#     # We need to be able to handle data across multiple data files.
#     # So build a list of all the files and pull out relevant info.  
#     reg = [ re.search('(\w+)_(\w+)_(\w+)_(\w+).dat',fname) for fname in filelist];
# 
#     # Okay, lets get all the different datafiles into a structure we can use.
#     # Initialize the data dictionary.
#     data = dict((('clusters',None), ('onsets',None), ('sync',None), 
#         ('bhv',None)));
# 
#     # Populate the dictionary with our timing, behavior, and spiking information
#     for ii in np.arange(len(reg)):
#         if reg[ii] != None:
#             if reg[ii].group(1) == 'clusters':
#                 fin = open(datadir + reg[ii].group(0),'r')
#                 data['clusters'] = pkl.load(fin)
#                 fin.close()
#             
#             elif reg[ii].group(1) == 'onsets':
#                 fin = open(datadir + reg[ii].group(0),'r')
#                 data['onsets'] = pkl.load(fin)
#                 fin.close()
#         
#             elif reg[ii].group(1) == 'sync':
#                 fin = open(datadir + reg[ii].group(0),'r')
#                 data['sync'] = pkl.load(fin)
#                 fin.close()
#             
#             elif reg[ii].group(1) == 'bhv':
#                 fin = open(datadir + reg[ii].group(0),'r')
#                 data['bhv'] = pkl.load(fin)
#                 fin.close()
#             
#             rat = reg[ii].group(2);
#             date = reg[ii].group(3);
        
    tetrode = 2

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
    
    #~ Need to build a big list of each trial.  For each trial, here is what we need:
       
        #~ Previous goal (PG) port
        #~ Future goal (FG) port
        #~ Previous outcome
        #~ Future outcome
        #~ PG time
        #~ FG time
        #~ The rat's response (right or left)
        #~ All the spikes PG time to FG time

    #~ Probably more, I'll list more as I think of them

    
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
    
def raster(trials, trial_spikes):
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
    
    sorting['indices'] = range(0,len(trials))
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
    
def LLRR_figs(trials, spikes, bin_width = 0.2, range = (-7,1)):
    """ Makes raster and histogram plots for each trajectory, cued and 
    uncued."""
    
    consts = constants()
    
    # Find the trials for all four PG-FG combinations, in the uncued blocks,
    # and FG hits only.
    
    right_left = by_condition(trials, 'right', 'left', 'uncued', 'hit')
    left_left = by_condition(trials, 'left', 'left', 'uncued', 'hit')
    left_right = by_condition(trials, 'left', 'right', 'uncued', 'hit')
    right_right = by_condition(trials, 'right', 'right', 'uncued', 'hit')
        
    # Grab the spikes for each trial that satisfies the conditions
    
    spks_l_l = [ spikes[ind] for ind in left_left ]
    spks_l_r = [ spikes[ind] for ind in left_right ]
    spks_r_r = [ spikes[ind] for ind in right_right ]
    spks_r_l = [ spikes[ind] for ind in right_left ]
    
    ylims=[0]*4
    
    # Now plot all four on one figure
    plt.figure()
    plt.subplot(221)
    peth(trials[right_left], spks_r_l, bin_width, range, 
        label = 'Right->left' + " n = " + str(len(spks_r_l)));
    plt.title('Right->left, uncued, FG hits')
    plt.xlim(range)
    ylims[0] = plt.ylim()[1]
    
    plt.subplot(222)
    peth(trials[left_left], spks_l_l, bin_width, range, 
        label = 'Left->left' + " n = " + str(len(spks_l_l)));
    plt.title('Left->left')
    plt.xlim(range)
    ylims[1] = plt.ylim()[1]
    
    plt.subplot(223)
    peth(trials[left_right], spks_l_r, bin_width, range, 
        label = 'Left->right' + " n = " + str(len(spks_l_r)));
    plt.title('Left->right')
    plt.xlim(range)
    ylims[2] = plt.ylim()[1]
    
    plt.subplot(224)
    peth(trials[right_right], spks_r_r, bin_width, range, 
        label = 'Right->right' + " n = " + str(len(spks_r_r)));
    plt.title('Right->right')
    plt.xlim(range)
    ylims[3] = plt.ylim()[1]
    
    # I want to set the y-scale equal for all four plots.
    ymax = np.max(ylims)
    
    plt.subplot(221)
    plt.ylim(0,ymax)
    
    plt.subplot(222)
    plt.ylim(0,ymax)
    
    plt.subplot(223)
    plt.ylim(0,ymax)
    
    plt.subplot(224)
    plt.ylim(0,ymax)
    
    #plt.legend(loc='upper left')
    #plt.title('Uncued, FG hits')
    
    # Let's also look at the raster plots
    
    plt.figure()
    
    plt.subplot(221)
    raster(trials[right_left], spks_r_l)
    plt.title('Right->left')
    plt.xlim(range)
    
    plt.subplot(222)
    raster(trials[left_left], spks_l_l)
    plt.title('Left->left, uncued, FG hits')
    plt.xlim(range)
    
    plt.subplot(223)
    raster(trials[left_right], spks_l_r)
    plt.title('Left->right')
    plt.xlim(range)
    
    plt.subplot(224)
    raster(trials[right_right], spks_r_r)
    plt.title('Right->right')
    plt.xlim(range)
    
    
    
    #########################################################################
    
    # Cued blocks this time.
    
    # Find the trials for all four PG-FG combinations, in the cued and uncued blocks,
    # and FG hits only.
    
    right_left = by_condition(trials, 'right', 'left', 'cued', 'hit')
    left_left = by_condition(trials, 'left', 'left', 'cued', 'hit')
    left_right = by_condition(trials, 'left', 'right', 'cued', 'hit')
    right_right = by_condition(trials, 'right', 'right', 'cued', 'hit')
        
    # Grab the spikes for each trial that satisfies the conditions
    
    spks_l_l = [ spikes[ind] for ind in left_left ]
    spks_l_r = [ spikes[ind] for ind in left_right ]
    spks_r_r = [ spikes[ind] for ind in right_right ]
    spks_r_l = [ spikes[ind] for ind in right_left ]
    
    
    # Now plot all four on one figure
    plt.figure()
    
    ylims = [0]*4
    
    plt.subplot(221)
    peth(trials[right_left], spks_r_l, bin_width, range, 
        label = 'Right->left' + " n = " + str(len(spks_r_l)));
    plt.title('Right->left, cued, FG hits')
    plt.xlim(range)
    ylims[0]=plt.ylim()[1]
    
    plt.subplot(222)
    peth(trials[left_left], spks_l_l, bin_width, range, 
        label = 'Left->left' + " n = " + str(len(spks_l_l)));
    plt.title('Left->left')
    plt.xlim(range)
    ylims[1]=plt.ylim()[1]
    
    plt.subplot(223)
    peth(trials[left_right], spks_l_r, bin_width, range, 
        label = 'Left->right' + " n = " + str(len(spks_l_r)));
    plt.title('Left->right')
    plt.xlim(range)
    ylims[2]=plt.ylim()[1]
    
    plt.subplot(224)
    peth(trials[right_right], spks_r_r, bin_width, range, 
        label = 'Right->right' + " n = " + str(len(spks_r_r)));
    plt.title('Right->right')
    plt.xlim(range)
    ylims[3]=plt.ylim()[1]
    
    
    # I want to set the y-scale equal for all four plots.
    ymax = np.max(ylims)
    
    plt.subplot(221)
    plt.ylim(0,ymax)
    
    plt.subplot(222)
    plt.ylim(0,ymax)
    
    plt.subplot(223)
    plt.ylim(0,ymax)
    
    plt.subplot(224)
    plt.ylim(0,ymax)
    
    #plt.legend(loc='upper left')
    #plt.title('Cued, FG hits')
    
    # Let's also look at the raster plots
    
    plt.figure()
    
    plt.subplot(221)
    raster(trials[right_left], spks_r_l)
    plt.title('Right->left')
    plt.xlim(range)
    
    plt.subplot(222)
    raster(trials[left_left], spks_l_l)
    plt.title('Left->left, cued, FG hits')
    plt.xlim(range)
    
    plt.subplot(223)
    raster(trials[left_right], spks_l_r)
    plt.title('Left->right')
    plt.xlim(range)
    
    plt.subplot(224)
    raster(trials[right_right], spks_r_r)
    plt.title('Right->right')
    plt.xlim(range)
    
def cue_uncue(trials, spikes, bin_width = 0.2, range = (-7,1)):
    ''' This methods makes plots comparing cued and uncued PETHs '''
    
    conditions = [0]*8
    conditions[0] = by_condition(trials, 'right', 'left', 'cued', 'hit')
    conditions[1] = by_condition(trials, 'right', 'left', 'uncued', 'hit')
    conditions[2] = by_condition(trials, 'left', 'left', 'cued', 'hit')
    conditions[3] = by_condition(trials, 'left', 'left', 'uncued', 'hit')
    conditions[4] = by_condition(trials, 'left', 'right', 'cued', 'hit')
    conditions[5] = by_condition(trials, 'left', 'right', 'uncued', 'hit')
    conditions[6] = by_condition(trials, 'right', 'right', 'cued', 'hit')
    conditions[7] = by_condition(trials, 'right', 'right', 'uncued', 'hit')
    
    c_spikes = [0]*8
    for ii, cond in enumerate(conditions):
        c_spikes[ii] = [ spikes[ind] for ind in cond ]
    
    # Now plot all four on one figure
    plt.figure()
    
    ylims = [0]*4
    labels = ['Right->left', 'Left->left', 'Left->right', 'Right->right']

    for ii in np.arange(4): 
        plt.subplot(220 + ii + 1)
        
        # Plot averaged PETHs
        x, unavg, uncued = peth(trials[conditions[2*ii+1]], c_spikes[2*ii+1],
            bin_width, range, label = 'Uncued')
        x, cuavg, cued = peth(trials[conditions[2*ii]], c_spikes[2*ii],
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
    

    


def bar_figs(trials, spikes):
    # Make some bar plots here
    _bar_fig(trials, spikes, 'PG in', 'PG out', title = 'In PG port')
    _bar_fig(trials, spikes, 'PG out', 'Center in', title = 'PG out to center in')
    _bar_fig(trials, spikes, 'Center in', 'Center out', title = 'In center port')
    _bar_fig(trials, spikes, 'Center out', 'FG in', title = 'Center out to FG in')
    _bar_fig(trials, spikes, 'PG in', 'Response', title = 'PG in to response')


def _bar_fig(trials, spikes, low_event, high_event, title = None):
    ''' This method will make some figures, in particular, bar plots.  I want
    to compare firing rates during the delay period between all my conditions.
    So, the plots will compare average firing rate during the delay period, 
    that is between the previous goal and response.
    
    Arguments:
    trials : structured array storing trials information, returned from 
        timelock function
    spikes : list of spike times for each trial
    low_event : the event 
    '''
    
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
    
    # Calculate average firing rates between previous goal and response
    
    means = []
    
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
        
        #means.append(np.mean(trial[(x > pg) & (x < rs)]))
        
        try:
            num = len(spks[(spks > low) & (spks < high)])
        except:
            num = 0
        means.append((num / np.abs(high - low))*trials['Scale'][ii][0])
        
        #means.append(len(spikes[( x>pg ) & (x < rs)])/np.abs((rs - pg)))
    
    means = np.array(means)
    
    # Now let's make some bar plots.
    # First plot will be for each trajectory, compare between cued and uncued
    
    conditions = [0]*8
    
    # Uncued trials
    conditions[0] = by_condition(trials, 'right', 'left', 'uncued', 'hit')
    conditions[1] = by_condition(trials, 'left', 'left', 'uncued', 'hit')
    conditions[2] = by_condition(trials, 'left', 'right', 'uncued', 'hit')
    conditions[3] = by_condition(trials, 'right', 'right', 'uncued', 'hit')
    
    # Cued trials
    conditions[4] = by_condition(trials, 'right', 'left', 'cued', 'hit')
    conditions[5] = by_condition(trials, 'left', 'left', 'cued', 'hit')
    conditions[6] = by_condition(trials, 'left', 'right', 'cued', 'hit')
    conditions[7] = by_condition(trials, 'right', 'right', 'cued', 'hit')
    
    # We'll get the means and standard errors for each bar
    
    heights, errs = np.zeros(len(conditions)), np.zeros(len(conditions))
    for ii, cond in enumerate(conditions):
        heights[ii] = np.mean(means[cond])
        errs[ii] = np.std(means[cond])/np.sqrt(len(means[cond]))
    
    # Now let's check signifcance, using Mann-Whitney U
    # We want to check each pair
    sigs = dict()
    itercomb = combinations(np.arange(0,8),2)
    for ii, jj in itercomb:
        samp1 = means[conditions[ii]]
        samp2 = means[conditions[jj]]
        sigtest = ut.ranksum_small(samp1, samp2)[1]
        sig = sigtest <= 0.05
        sigs.update({(ii,jj) : sig})

    # Now plot!  Bar plots!
    plt.figure()
    
    left_u = range(1,12,3)
    plt.bar(left_u, heights[0:4], width= 1, yerr = errs[0:4], color = 'b', 
        label = 'Uncued')
    
    left_c = range(2,13,3)
    plt.bar(left_c, heights[4:8], width= 1, yerr = errs[4:8], color = 'g',
        label = 'Cued')
    
    # Now let's plot some significance information
    itercomb = combinations(np.arange(0,8),2)
    for ii, jj in itercomb:
        # We only care about comparing uncued to cued for the same trajectory,
        # and uncued between trajectories
        
        if (jj-ii == 4) or ((ii in np.arange(0,4)) & (jj in np.arange(0,4))):
            pass
        else:
            continue
        
        signif = sigs[(ii,jj)]
        if signif:
            # If significant, plot an asterisk and some lines 
            
            # Set the left x location
            if ii in np.arange(0,4):
                lx = left_u[ii] + 0.5
            else:
                lx = left_c[ii-4] + 0.5
            # Set the right x location
            if jj in np.arange(0,4):
                rx = left_u[jj] + 0.5
            else:
                rx = left_c[jj-4] + 0.5
            
            # Find how high the bars are
            l_barh = heights[ii] + errs[ii]
            r_barh = heights[jj] + errs[jj]
            barh = max(heights) + max(errs)
            
            # Set the horizontal line height
            if (ii in np.arange(1,5)) & (jj in np.arange(5,9)):
                lineh = barh + 0.2*barh*(np.random.randn()+1)
            else:
                lineh = barh + 0.1*barh*(np.random.randn()+1)

            # Plot the two vertical lines above the bars
            plt.plot([lx, lx], [l_barh + l_barh*0.05, lineh], '-k')
            plt.plot([rx, rx], [r_barh + r_barh*0.05, lineh], '-k')
            
            # Plot the horizontal line
            plt.plot([lx, rx], [lineh, lineh], '-k')
    
    plt.ylabel('Activity (Hz)')
    plt.xlabel('Condition')
    plt.xticks(range(2,12,3), 
        ('right->left', 'left->left', 'left->right', 'right->right'))
    plt.xlim(0,13)
    plt.legend()
    
    if title != None:
        plt.title(title)
       
    plt.show() 
    

def rate_by_trial(trials, spikes, low_event, high_event):
    
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
    
    # Calculate average firing rates between previous goal and response
    x, avg_all, by_trial = peth(trials, spikes, range = (-20, 3), to_plot=0)
    
    n_spikes = by_trial/5.0
    
    spike_count = [0]*len(by_trial)

    for ii, spks in enumerate(n_spikes):
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
        
        limits = ut.find((x>=low) & (x<=high))
        
        count = np.sum(spks[limits])
        spike_count[ii] = count
    
    events, spks = np.histogram(spike_count, bins = 40, range = (0,40))
    
    plt.plot(spks[:-1], events)
    plt.show()
    return spike_count

def 
        

def by_condition(trials,  PG = 'all', FG = 'all', block = 'all',
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
    
    return np.sort(list(trial_set))

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
