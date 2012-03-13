""" Okay, this one is going to take the behavior and neural data then generate
timelocked raster plots and peths (post stimulus time histograms).  Maybe I
should actually call them Pre Response Time Histograms.  I'll figure it out.
"""

import os
import re
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.io

def timelock_RS(datadir, n_shift = 0, norm = None):
    """ This function takes the behavior and spiking data, then returns
    the spikes timelocked to the 'Response,' when the stimulus plays.
    
    Arguments:
    datadir : path to the directory where the data files are stored.
        The necessary data files are behavior, clusters, syncing, and onsets from
        the recording data.
    
    n_shift : Sometimes the data will be split into multiple files.  If this happens,
    this parameter will shift everything forward trial to trial number n_shift.
    """
    
    
    # First we need to import the behavior and neural data, also the syncing
    # information.

    # The path to where the data is stored.
    #datadir = '/home/mat/Dropbox/Data/'

    # Get list of files in datadir
    filelist = os.listdir(datadir);
    filelist.sort();

    # We need to be able to handle data across multiple data files.
    # So build a list of all the files and pull out relevant info.  
    reg = [ re.search('(\w+)_(\w+)_(\w+)_(\w+).dat',fname) for fname in filelist];

    
    # Okay, lets get all the different datafiles into a structure we can use.

    data_dict = dict((('clusters',None),('onsets',None),('sync',None),('behave',None)));

    for ii in np.arange(len(reg)):
        if reg[ii] != None:
            if reg[ii].group(1) == 'clusters':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['clusters'] = pkl.load(fin)
                fin.close()
            
            elif reg[ii].group(1) == 'onsets':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['onsets'] = pkl.load(fin)
                fin.close()
        
            elif reg[ii].group(1) == 'sync':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['sync'] = pkl.load(fin)
                fin.close()
            
            elif reg[ii].group(1) == 'behave':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['behave'] = pkl.load(fin)
                fin.close()
            
            rat = reg[ii].group(2);
            date = reg[ii].group(3);

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
    b_onsets = data_dict['behave']['onsets'];
    consts = data_dict['behave']['CONSTS'];
    
    # Create a structured array to keep all the relevant trial information
    records = [('PG port','i8'), ('FG port','i8'), ('PG outcome','i8'), \
        ('FG outcome','i8'), ('PG time','f8'), ('FG time', 'f8'), \
        ('PG response','i8'), ('Response','i8'), ('Trial length', 'f8'), \
        ('Scale', 'f8', 2)];

    trials = np.zeros((len(b_onsets),),dtype = records)

    # Populate the array

    # Correct port for the previous goal
    trials['PG port'] = np.concatenate((np.array((0,)), \
        data_dict['behave']['TRIALS_INFO']['CORRECT_SIDE'][:len(b_onsets)-1]));

    # Correct port for the future goal
    trials['FG port'] = data_dict['behave']['TRIALS_INFO']['CORRECT_SIDE'][:len(b_onsets)];

    # Was the previous response a hit or an error
    trials['PG outcome'] = np.concatenate((np.array((0,)), \
        data_dict['behave']['TRIALS_INFO']['OUTCOME'][:len(b_onsets)-1]))

    # Was the future response a hit or an error
    trials['FG outcome'] = data_dict['behave']['TRIALS_INFO']['OUTCOME'][:len(b_onsets)]
    
    # What direction did the rat go (left = 1, right = 2)
    for ii in np.arange(len(b_onsets)):
        if trials['FG outcome'][ii] == consts['HIT']:
            trials['Response'][ii] = trials['FG port'][ii]
        elif trials['FG outcome'][ii] == consts['ERROR']:
            if trials['FG port'][ii] == consts['LEFT']:
                trials['Response'][ii] = consts['RIGHT']
            elif trials['FG port'][ii] == consts['RIGHT']:
                trials['Response'][ii] = consts['LEFT']
    
    trials['PG response'] = np.concatenate((np.array((0,)),trials['Response'][:-1]));
    
    trials['Scale'] = np.ones((len(trials),2));
    
    # Now we need to get the times of previous and future goals.  I'm going to timelock
    # to the stimulus, i.e. the response, so it will be relative to the onset time.

    # Previous goal time is defined as when the rat pokes in a side port on the
    # previous trial.

    # Future goal time is defined as when the rat pokes in a side port on the current trial.
    for ii in np.arange(len(b_onsets)):
        
        times = data_dict['behave']['peh'][ii];
        
        # Get previous goal times
        if ii > 0:
            prev_times = data_dict['behave']['peh'][ii-1];
            
            if trials['PG outcome'][ii] == consts['HIT']:
                pg_time = prev_times['states']['hit_istate'].min();
            elif trials['PG outcome'][ii] == consts['ERROR']:
                pg_time = prev_times['states']['error_istate'].min();
            elif trials['PG outcome'][ii] == consts['CHOICE_TIME_UP']:
                pg_time = prev_times['states']['choice_time_up_istate'].min();
        else:
            pg_time = b_onsets[ii];
        
        trials['PG time'][ii] = pg_time - b_onsets[ii];
            
        # Now get the future goal times
        if trials['FG outcome'][ii] == consts['HIT']:
            fg_time = times['states']['hit_istate'].min();
        elif trials['FG outcome'][ii] == consts['ERROR']:
            fg_time = times['states']['error_istate'].min();
        elif trials['FG outcome'][ii] == consts['CHOICE_TIME_UP']:
            fg_time = times['states']['choice_time_up_istate'].min();
        
        trials['FG time'][ii] = fg_time - b_onsets[ii];
        
        trials['Trial length'][ii] = fg_time - pg_time;
    
    
    # Get the scaling factors
    if norm == 'PG':
        for jj in np.arange(len(trials)):
            scale = -7./trials['PG time'][jj];
            trials['Scale'][jj] = np.array((scale,1));
    elif norm == 'FG':
        for jj in np.arange(len(trials)):
            scale = 1./trials['FG time'][jj];
            trials['Scale'][jj] = np.array((1,scale));
    elif norm == 'PG + FG':
        for jj in np.arange(len(trials)):
            scale_neg = -7./trials['PG time'][jj];
            scale_pos = 1./trials['FG time'][jj];
            trials['Scale'][jj] = np.array((scale_neg, scale_pos));
            
    
    
    # Okay, now get the spikes and time lock to the response, which we are
    # defining as the center poke, the stimulus onset.

    # Let's first sync up the behavior trials and recording onsets

    # If there were more than one neural data files, then you might have
    # to shift things some.  Set this to the number of behavior trials 
    # skipped before the recording data starts.
    #n_shift = 24;

    sync = data_dict['sync'].map_n_to_b;

    n_onsets = data_dict['onsets']/30000.

    trials_spikes = []

    # Loop over each cluster in the ePhys data
    for cl in data_dict['clusters']:
        spikes = [0]*len(b_onsets);
        
        p_times = cl['peaks'];
        
        if n_shift != None:
            shift = 1
        else:
            shift = 0
        
        # Go through each trial and grab the spikes for that trial
        for ii in sync - shift:
            
            # Find the spikes between the PG and FG (plus 3 sec on both sides)
            ind = (p_times > n_onsets[ii] + trials['PG time'][ii+n_shift] - 30.) & \
                (p_times < n_onsets[ii] + trials['FG time'][ii+n_shift] + 30);
            
            spikes[ii+n_shift] = p_times[ind]-n_onsets[ii];
            spikes[ii+n_shift].sort();
            
        if norm == 'PG':
            for jj in np.arange(len(trials)):
                spikes[jj] = trials['Scale'][jj][0] * spikes[jj];
        elif norm == 'FG':
            for jj in np.arange(len(trials)):
                spikes[jj] = trials['Scale'][jj][1] * spikes[jj];
        elif norm == 'PG + FG':
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
       
    if norm == 'PG':
        for jj in np.arange(len(trials)):
            if jj == 0:
                trials['PG time'][jj] = 0;
            else:
                trials['PG time'][jj] = trials['Scale'][jj][0] * trials['PG time'][jj];
    elif norm == 'FG':
        for jj in np.arange(len(trials)):
            trials['FG time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
    elif norm == 'PG + FG':
        for jj in np.arange(len(trials)):
            if jj == 0:
                trials['PG time'][jj] = 0;
            else:
                trials['PG time'][jj] = trials['Scale'][jj][0] * trials['PG time'][jj];
            trials['FG time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
     
    
    # Exclude trials longer than 30 seconds
    th = trials['Trial length'] < 30.
    trials = trials[th]
    
    for x,spikes in enumerate(trials_spikes):
        trials_spikes[x] = [ spikes[ii] for ii in (np.nonzero(th)[0] - n_shift) ]
       
    
    return trials[n_shift:], trials_spikes
    
def timelock_PG(datadir, n_shift = 0, norm = None):
    """ This function takes the behavior and spiking data, then returns
    the spikes timelocked to the previous goal, when the rat gets the reward on the previous trial.
    
    Arguments:
    datadir : path to the directory where the data files are stored.
        The necessary data files are behavior, clusters, syncing, and onsets from
        the recording data.
    
    n_shift : Sometimes the data will be split into multiple files.  If this happens,
    this parameter will shift everything forward n_shift trials.
    """
    
    # First we need to import the behavior and neural data, also the syncing
    # information.

    # The path to where the data is stored.
    #datadir = '/home/mat/Dropbox/Data/'

    # Get list of files in datadir
    filelist = os.listdir(datadir);
    filelist.sort();

    # We need to be able to handle data across multiple data files.
    # So build a list of all the files and pull out relevant info.  
    reg = [ re.search('(\w+)_(\w+)_(\w+)_(\w+).dat',fname) for fname in filelist];

    
    # Okay, lets get all the different datafiles into a structure we can use.

    data_dict = dict((('clusters',None),('onsets',None),('sync',None),('behave',None)));

    for ii in np.arange(len(reg)):
        if reg[ii] != None:
            if reg[ii].group(1) == 'clusters':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['clusters'] = pkl.load(fin)
                fin.close()
            
            elif reg[ii].group(1) == 'onsets':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['onsets'] = pkl.load(fin)
                fin.close()
        
            elif reg[ii].group(1) == 'sync':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['sync'] = pkl.load(fin)
                fin.close()
            
            elif reg[ii].group(1) == 'behave':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['behave'] = pkl.load(fin)
                fin.close()
            
            rat = reg[ii].group(2);
            date = reg[ii].group(3);

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
    b_onsets = data_dict['behave']['onsets']
    consts = data_dict['behave']['CONSTS']

    # Create a structured array to keep all the relevant trial information
    records = [('PG port','i8'), ('FG port','i8'), ('PG outcome','i8'), \
        ('FG outcome','i8'), ('RS time','f8'), ('FG time', 'f8'), \
        ('PG response','i8'), ('Response','i8'), ('Trial length','f8'),\
        ('Scale', 'f8', 2)];

    trials = np.zeros((len(b_onsets),),dtype = records)

    # Populate the array

    # Correct port for the previous goal
    trials['PG port'] = np.concatenate((np.array((0,)), \
        data_dict['behave']['TRIALS_INFO']['CORRECT_SIDE'][:len(b_onsets)-1]));

    # Correct port for the future goal
    trials['FG port'] = data_dict['behave']['TRIALS_INFO']['CORRECT_SIDE'][:len(b_onsets)];

    # Was the previous response a hit or an error
    trials['PG outcome'] = np.concatenate((np.array((0,)), \
        data_dict['behave']['TRIALS_INFO']['OUTCOME'][:len(b_onsets)-1]))

    # Was the future response a hit or an error
    trials['FG outcome'] = data_dict['behave']['TRIALS_INFO']['OUTCOME'][:len(b_onsets)]
    
    # What direction did the rat go (left = 1, right = 2)
    for ii in np.arange(len(b_onsets)):
        if trials['FG outcome'][ii] == consts['HIT']:
            trials['Response'][ii] = trials['FG port'][ii]
        elif trials['FG outcome'][ii] == consts['ERROR']:
            if trials['FG port'][ii] == consts['LEFT']:
                trials['Response'][ii] = consts['RIGHT']
            elif trials['FG port'][ii] == consts['RIGHT']:
                trials['Response'][ii] = consts['LEFT']
    
    trials['PG response'] = np.concatenate((np.array((0,)),trials['Response'][:-1]));
    
    # Now we need to get the times of previous and future goals.  I'm going to timelock
    # to the stimulus, i.e. the response, so it will be relative to the onset time.

    # Previous goal time is defined as when the rat pokes in a side port on the
    # previous trial.

    # Future goal time is defined as when the rat pokes in a side port on the current trial.
    for ii in np.arange(len(b_onsets)):
        
        times = data_dict['behave']['peh'][ii];
        
        # Get previous goal times
        if ii > 0:
            prev_times = data_dict['behave']['peh'][ii-1];
            
            if trials['PG outcome'][ii] == consts['HIT']:
                pg_time = prev_times['states']['hit_istate'].min();
            elif trials['PG outcome'][ii] == consts['ERROR']:
                pg_time = prev_times['states']['error_istate'].min();
            elif trials['PG outcome'][ii] == consts['CHOICE_TIME_UP']:
                pg_time = prev_times['states']['choice_time_up_istate'].min();
        else:
            pg_time = b_onsets[ii];
        
        
        trials['RS time'][ii] = b_onsets[ii] - pg_time;
            
        # Now get the future goal times
        if trials['FG outcome'][ii] == consts['HIT']:
            fg_time = times['states']['hit_istate'].min();
        elif trials['FG outcome'][ii] == consts['ERROR']:
            fg_time = times['states']['error_istate'].min()min();
        elif trials['FG outcome'][ii] == consts['CHOICE_TIME_UP']:
            fg_time = times['states']['choice_time_up_istate'].min();
        
        trials['FG time'][ii] = fg_time - pg_time;
        
        trials['Trial length'][ii] = fg_time - pg_time;
        
    # Get the scaling factors
    trials['Scale'] = np.ones((len(trials),2));
    
    if norm == 'RS':
        for jj in np.arange(len(trials)):
            scale = 7./trials['RS time'][jj];
            trials['Scale'][jj] = np.array((scale,1));
    elif norm == 'FG':
        for jj in np.arange(len(trials)):
            scale = 8./trials['FG time'][jj];
            trials['Scale'][jj] = np.array((1,scale));
    elif norm == 'RS + FG':
        for jj in np.arange(len(trials)):
            scale_neg = 7./trials['RS time'][jj];
            scale_pos = 8./trials['FG time'][jj];
            trials['Scale'][jj] = np.array((scale_neg, scale_pos));
    
    # Okay, now get the spikes and time lock to the response, which we are
    # defining as the center poke, the stimulus onset.

    # Let's first sync up the behavior trials and recording onsets

    # If there were more than one neural data files, then you might have
    # to shift things some.  Set this to the number of behavior trials 
    # skipped before the recording data starts.
    #n_shift = 24;

    sync = data_dict['sync'].map_n_to_b;

    n_onsets = data_dict['onsets']/30000.

    trials_spikes = []

    # Loop over each cluster in the ePhys data
    for cl in data_dict['clusters']:
        spikes = [0]*len(b_onsets);
        
        p_times = cl['peaks'];
        
        # Go through each trial and grab the spikes for that trial
        for ii in sync:
            
            # Find the spikes between the PG and FG (plus 3 sec on both sides)
            ind = (p_times > n_onsets[ii] - trials['RS time'][ii+n_shift] - 30.) & \
                (p_times < n_onsets[ii] + trials['FG time'][ii+n_shift] - 
                    trials['RS time'][ii+n_shift] + 30);
            
            spikes[ii+n_shift] = p_times[ind] - n_onsets[ii] + trials['RS time'][ii+n_shift];
            spikes[ii+n_shift].sort();
            
        # This part scales the spike times
        if norm == 'RS':
            for jj in np.arange(len(trials)):
                spikes[jj] = trials['Scale'][jj][0] * spikes[jj];
        elif norm == 'FG':
            for jj in np.arange(len(trials)):
                spikes[jj] = trials['Scale'][jj][1] * spikes[jj];
        elif norm == 'RS + FG':
            for jj in np.arange(len(trials)):
                try: 
                    spikes[jj][spikes[jj]<7] = trials['Scale'][jj][0] * spikes[jj][spikes[jj]<7];
                except:
                    spikes[jj] = spikes[jj];
                try:
                    spikes[jj][spikes[jj]>7] = trials['Scale'][jj][1] * spikes[jj][spikes[jj]>7];
                except:
                    spikes[jj] = spikes[jj];
                
            
        trials_spikes.append(spikes[n_shift:])
    
    # This part scales all the PG and FG times    
    if norm == 'RS':
        for jj in np.arange(len(trials)):
            trials['RS time'][jj] = trials['Scale'][jj][0] * trials['RS time'][jj];
    elif norm == 'FG':
        for jj in np.arange(len(trials)):
            trials['FG time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
    elif norm == 'RS + FG':
        for jj in np.arange(len(trials)):
            trials['RS time'][jj] = trials['Scale'][jj][0] * trials['RS time'][jj];
            trials['FG time'][jj] = trials['Scale'][jj][1] * trials['FG time'][jj];
       
    # Exclude trials longer than 30 seconds
    th = trials['Trial length'] < 30.
    trials = trials[th]
    
    for x,spikes in enumerate(trials_spikes):
        trials_spikes[x] = [ spikes[ii] for ii in np.nonzero(th)[0] ]
    
    return trials[n_shift:], trials_spikes

def timelock_FG(datadir, n_shift = 0, norm = None):
    """ This function takes the behavior and spiking data, then returns
    the spikes timelocked to the previous goal, when the rat gets the reward on the previous trial.
    
    Arguments:
    datadir : path to the directory where the data files are stored.
        The necessary data files are behavior, clusters, syncing, and onsets from
        the recording data.
    
    n_shift : Sometimes the data will be split into multiple files.  If this happens,
    this parameter will shift everything forward n_shift trials.
    """
    
    # First we need to import the behavior and neural data, also the syncing
    # information.

    # The path to where the data is stored.
    #datadir = '/home/mat/Dropbox/Data/'

    # Get list of files in datadir
    filelist = os.listdir(datadir);
    filelist.sort();

    # We need to be able to handle data across multiple data files.
    # So build a list of all the files and pull out relevant info.  
    reg = [ re.search('(\w+)_(\w+)_(\w+)_(\w+).dat',fname) for fname in filelist];

    
    # Okay, lets get all the different datafiles into a structure we can use.

    data_dict = dict((('clusters',None),('onsets',None),('sync',None),('behave',None)));

    for ii in np.arange(len(reg)):
        if reg[ii] != None:
            if reg[ii].group(1) == 'clusters':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['clusters'] = pkl.load(fin)
                fin.close()
            
            elif reg[ii].group(1) == 'onsets':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['onsets'] = pkl.load(fin)
                fin.close()
        
            elif reg[ii].group(1) == 'sync':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['sync'] = pkl.load(fin)
                fin.close()
            
            elif reg[ii].group(1) == 'behave':
                fin = open(datadir + reg[ii].group(0),'r')
                data_dict['behave'] = pkl.load(fin)
                fin.close()
            
            rat = reg[ii].group(2);
            date = reg[ii].group(3);

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
    b_onsets = data_dict['behave']['onsets']
    consts = data_dict['behave']['CONSTS']

    # Create a structured array to keep all the relevant trial information
    records = [('PG port','i8'), ('FG port','i8'), ('PG outcome','i8'), \
        ('FG outcome','i8'), ('PG time','f8'), ('RS time', 'f8'), \
        ('PG response','i8'), ('Response','i8'), ('Trial length', 'f8'),\
        ('Scale', 'f8', 2)];

    trials = np.zeros((len(b_onsets),),dtype = records)

    # Populate the array

    # Correct port for the previous goal
    trials['PG port'] = np.concatenate((np.array((0,)), \
        data_dict['behave']['TRIALS_INFO']['CORRECT_SIDE'][:len(b_onsets)-1]));

    # Correct port for the future goal
    trials['FG port'] = data_dict['behave']['TRIALS_INFO']['CORRECT_SIDE'][:len(b_onsets)];

    # Was the previous response a hit or an error
    trials['PG outcome'] = np.concatenate((np.array((0,)), \
        data_dict['behave']['TRIALS_INFO']['OUTCOME'][:len(b_onsets)-1]))

    # Was the future response a hit or an error
    trials['FG outcome'] = data_dict['behave']['TRIALS_INFO']['OUTCOME'][:len(b_onsets)]
    
    # What direction did the rat go (left = 1, right = 2)
    for ii in np.arange(len(b_onsets)):
        if trials['FG outcome'][ii] == consts['HIT']:
            trials['Response'][ii] = trials['FG port'][ii]
        elif trials['FG outcome'][ii] == consts['ERROR']:
            if trials['FG port'][ii] == consts['LEFT']:
                trials['Response'][ii] = consts['RIGHT']
            elif trials['FG port'][ii] == consts['RIGHT']:
                trials['Response'][ii] = consts['LEFT']
    
    trials['PG response'] = np.concatenate((np.array((0,)),trials['Response'][:-1]));
    
    # Now we need to get the times of previous and future goals.  I'm going to timelock
    # to the stimulus, i.e. the response, so it will be relative to the onset time.

    # Previous goal time is defined as when the rat pokes in a side port on the
    # previous trial.

    # Future goal time is defined as when the rat pokes in a side port on the current trial.
    for ii in np.arange(len(b_onsets)):
        
        times = data_dict['behave']['peh'][ii];
        
        # Now get the response times
        if trials['FG outcome'][ii] == consts['HIT']:
            fg_time = times['states']['hit_istate'].min();
        elif trials['FG outcome'][ii] == consts['ERROR']:
            fg_time = times['states']['error_istate'].min();
        elif trials['FG outcome'][ii] == consts['CHOICE_TIME_UP']:
            fg_time = times['states']['choice_time_up_istate'].min();
        
        trials['RS time'][ii] = b_onsets[ii] - fg_time;
        
        # Get previous goal times
        if ii > 0:
            prev_times = data_dict['behave']['peh'][ii-1];
            
            if trials['PG outcome'][ii] == consts['HIT']:
                pg_time = prev_times['states']['hit_istate'].min();
            elif trials['PG outcome'][ii] == consts['ERROR']:
                pg_time = prev_times['states']['error_istate'].min();
            elif trials['PG outcome'][ii] == consts['CHOICE_TIME_UP']:
                pg_time = prev_times['states']['choice_time_up_istate'].min();
        else:
            pg_time = b_onsets[ii];
            
        
        trials['PG time'][ii] = pg_time - fg_time;
            
        trials['Trial length'][ii] = fg_time - pg_time;
        
        
    # Get the scaling factors
    trials['Scale'] = np.ones((len(trials),2));
    
    if norm == 'PG':
        for jj in np.arange(len(trials)):
            scale = -8./trials['PG time'][jj];
            trials['Scale'][jj] = np.array((scale,1));
    elif norm == 'RS':
        for jj in np.arange(len(trials)):
            scale = -1./trials['RS time'][jj];
            trials['Scale'][jj] = np.array((1,scale));
    elif norm == 'PG + RS':
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

    sync = data_dict['sync'].map_n_to_b;

    n_onsets = data_dict['onsets']/30000.

    trials_spikes = []

    # Loop over each cluster in the ePhys data
    for cl in data_dict['clusters']:
        spikes = [0]*len(b_onsets);
        
        p_times = cl['peaks'];
        
        # Go through each trial and grab the spikes for that trial
        for ii in sync - 1:
            
            # Find the spikes between the PG and RS (plus 30 sec on both sides)
            ind = (p_times > n_onsets[ii] + trials['PG time'][ii+n_shift] - 30.) & \
                (p_times < n_onsets[ii] - trials['RS time'][ii+n_shift] + 30);
            
            spikes[ii+n_shift] = p_times[ind] - n_onsets[ii] + trials['RS time'][ii+n_shift];
            spikes[ii+n_shift].sort();
        
        # This part scales the spike times
        if norm == 'PG':
            for jj in np.arange(len(trials)):
                spikes[jj] = trials['Scale'][jj][0] * spikes[jj];
        elif norm == 'RS':
            for jj in np.arange(len(trials)):
                spikes[jj] = trials['Scale'][jj][1] * spikes[jj];
        elif norm == 'PG + RS':
            for jj in np.arange(len(trials)):
                try: 
                    spikes[jj][spikes[jj]<-1] = trials['Scale'][jj][0] * spikes[jj][spikes[jj]<-1];
                except:
                    spikes[jj] = spikes[jj];
                try:
                    spikes[jj][spikes[jj]>-1] = trials['Scale'][jj][1] * spikes[jj][spikes[jj]>-1];
                except:
                    spikes[jj] = spikes[jj];
        
        trials_spikes.append(spikes[n_shift:])
       
    # This part scales all the PG and FG times    
    if norm == 'PG':
        for jj in np.arange(len(trials)):
            if jj == 0:
                trials['PG time'][jj] = -8;
            else:
                trials['PG time'][jj] = trials['Scale'][jj][0] * trials['PG time'][jj];
    elif norm == 'RS':
        for jj in np.arange(len(trials)):
            trials['RS time'][jj] = trials['Scale'][jj][1] * trials['RS time'][jj];
    elif norm == 'PG + RS':
        for jj in np.arange(len(trials)):
            if jj == 0:
                trials['PG time'][jj] = 0;
            else:
                trials['PG time'][jj] = trials['Scale'][jj][0] * trials['PG time'][jj];
            trials['RS time'][jj] = trials['Scale'][jj][1] * trials['RS time'][jj];
       
    # Exclude trials longer than 30 seconds
    th = trials['Trial length'] < 30.
    trials = trials[th]
    
    for x,spikes in enumerate(trials_spikes):
        trials_spikes[x] = [ spikes[ii] for ii in np.nonzero(th)[0] ]
    
    return trials[n_shift:], trials_spikes

def raster(trials, trial_spikes):
    plt.figure()

    # okay, each row is a different trial, each spikes should be marked 
    # as a vertical dash.
    ind = 1
    for ii in np.arange(len(trial_spikes)):
        
        spikes = trial_spikes[ii]
        
        if spikes.any():
            plt.plot(spikes, ind*np.ones((len(spikes),)),'|',color='k');
            
            if 'PG time' in trials.dtype.names:
                plt.plot(trials['PG time'][ii],ind,'.r');
            elif 'RS time' in trials.dtype.names:
                plt.plot(trials['RS time'][ii],ind,'.r');
            if 'FG time' in trials.dtype.names:
                plt.plot(trials['FG time'][ii],ind,'.b');
            elif 'RS time' in trials.dtype.names:
                plt.plot(trials['RS time'][ii],ind,'.b');
            ind = ind + 1;
        
    plt.plot([0,0],[0,ind],'grey')
    
    plt.show()
    
    return None
    
def peth(trials, data, bin_width = .2,range = (-10,3), label = None ):
    
    n_trials = len(data);
    
    bins = np.diff(range)[0]/bin_width;
    
    if trials['Scale'][0,0] != 1.:
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
    
    plt.plot(x,peth, label = label);
    
    plt.show();
    
    return x, peth

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
    

def get_figures(trials, trial_spikes, bin_width = 0.2, range = (-7,2), \
    max_rate = 4, scaled = False):
    
    consts = constants();
    
    th = avg_rate(trial_spikes, to_plot = 0) < max_rate;
    trials = trials[th];
    trial_spikes =  [ trial_spikes[ii] for ii in th.nonzero()[0] ];

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
    peth(trials,spk_lefts, bin_width, range, label = 'left' + " n = " + str(len(spk_lefts)));
    peth(trials,spk_rights, bin_width, range, label = 'right'+ " n = " + str(len(spk_rights)));
    ax = fig1.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,max_rate + 1],'-',color = 'grey', lw=2)
    plt.ylim((0,max_rate));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig2 = plt.figure()
    peth(trials,spk_hits, bin_width, range, label = 'hits' + ' n = ' + str(len(spk_hits)));
    peth(trials,spk_errs, bin_width, range, label = 'errors' + ' n = ' + str(len(spk_errs)));
    ax = fig2.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,max_rate + 1],'-',color = 'grey', lw=2)
    plt.ylim((0,max_rate));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig3 = plt.figure()
    peth(trials,spk_pg_hits, bin_width, range, label = 'PG hits' + " n = " + str(len(spk_pg_hits)));
    peth(trials,spk_pg_errs, bin_width, range, label = 'PG errors' + " n = " + str(len(spk_pg_errs)));
    ax = fig3.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,max_rate + 1],'-',color = 'grey', lw=2)
    plt.ylim((0,max_rate));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig4 = plt.figure()
    peth(trials,spk_pg_left_hits, bin_width, range, label = 'PG left-hit' + " n = " + str(len(spk_pg_left_hits)));
    peth(trials,spk_pg_left_errs, bin_width, range, label = 'PG left-error' + " n = " + str(len(spk_pg_left_errs)));
    peth(trials,spk_pg_right_hits, bin_width, range, label = 'PG right-hit' + " n = " + str(len(spk_pg_right_hits)));
    peth(trials,spk_pg_right_errs, bin_width, range, label = 'PG right-error' + " n = " + str(len(spk_pg_right_errs)));
    ax = fig4.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,max_rate + 1],'-',color = 'grey', lw=2)
    plt.ylim((0,max_rate));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig5 = plt.figure()
    peth(trials,trial_spikes, bin_width, range, label = 'All trials' + " n = " + str(len(trial_spikes)));
    ax = fig5.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,max_rate + 1],'-',color = 'grey', lw=2)
    plt.ylim((0,max_rate));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    fig6 = plt.figure()
    peth(trials,spk_pg_lefts, bin_width, range, label = 'PG left' + " n = " + str(len(spk_pg_lefts)));
    peth(trials,spk_pg_rights, bin_width, range, label = 'PG right'+ " n = " + str(len(spk_pg_rights)));
    ax = fig6.gca();
    ax.legend(loc = 'upper left');
    plt.plot([0,0],[0,max_rate + 1],'-',color = 'grey', lw=2)
    plt.ylim((0,max_rate));
    plt.xlabel('Time (s)', size = 'x-large')
    plt.ylabel('Activity (spikes/s)', size = 'x-large')
    plt.xticks(size = 'large')
    plt.yticks(size = 'large')
    
    plt.show()
    
    
    
    
    
    
    
    