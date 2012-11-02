''' Rewriting timelock module probably.'''

import numpy as np
import bhv
import os
import re
import pickle as pkl
import bhv

def timelock(datadir, tetrode, zero = 'RS'):
    ''' temp '''
    
    
    
    ''' Sept 11, 2012, 6:37 PM:
        This function will align time stamped spikes to a behavioral event.
        First, it will look through a data directory containing four files, the
        files are behavior data, spiking data, onsets from the spike recording
        system, and syncing information between the behavior and recording
        onsets.
        
        So we'll load those files, then pull out behavior information using the
        Rat class from bhv.py.  From that data, we have the times of events
        which we can then use to set a time zero for each trial.  Based on that
        time zero, we'll shift the spike times appropriately.        
        '''
    
    filelist = os.listdir(datadir)
    filelist.sort()
    
    reg = [ re.search('(\w+)_(\w+).(\w+)', filename) for filename in filelist ]
    
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
    
    records = [('2PG port', 'i8'), ('PG port','i8'), ('FG port','i8'),
        ('2PG outcome', 'i8'), ('PG outcome','i8'), ('FG outcome','i8'),
        ('PG time','f8',2), ('RS time', 'f8'), ('FG time', 'f8'),
        ('C time', 'f8', 2), ('PG response','i8'), ('Response','i8'),
        ('Trial length', 'f8'), ('Block', 'i8'), ('Scale', 'f8', 2)]
    
    nTrials = len(data['ons'])
    trials = np.zeros(nTrials, dtype = records)
    
    bdata = data['bhv']['TRIALS_INFO']
    rat = bhv.Rat(name)
    rat.update(date, bdata)
    trialRecords = rat.sessions['trial_records']
    
    cat = np.concatenate()
    
    trials['FG port'] = trialRecords['correct side']
    
    trials['PG port'] = cat(np.zeros(1), trials['FG port'][:-1])
    
    trials['2PG port'] = cat(np.zeros(2), trials['FG port'][:-2])
    
    trials['FG outcome'] = trialRecords['outcome']
    
    trials['PG outcome'] = cat(np.zeros(1), trials['FG outcome'][:-1])
    
    trials['2PG outcome'] = cat(np.zeros(2), trials['FG outcome'][:-2])

    trials['Block'] = trialRecords['block']
    
    trials['Response'] = trialRecords['response']
    
    trials['PG response'] = cat(np.zeros(1), trials['Response'][:-1])

    trials['Scale'] = np.ones((nTrials,2))
    
    
    
    