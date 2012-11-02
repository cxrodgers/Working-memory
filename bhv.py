''' Behavior data analysis '''

import bcontrol
import numpy as np
from matplotlib.mlab import find

class Rat(object):
    
    def __init__(self, name):
        
        if not isinstance(name, str):
            raise TypeError, 'name should be a string'
        else:
            self.name = name
        self._data = {}
        self._session_records = [('n_trials', 'u2'), ('date', 'a6'), 
            ('performance', object), ('trials', object)]
        self.sessions = np.zeros(0, dtype = self._session_records)
    
    def update(self, date, bdata):
        
        if isinstance(date, str):
            place = True
            if date in self.sessions['date']:
                check = raw_input('Data exists for %s.  Replace data? [Yes/No] ' % date)
                while check not in ['Yes', 'No']:
                        check = raw_input('Data exists for %s.  Replace data? [Yes/No] ' 
                        		% date)
                if check == 'No':
                    place = False
                elif check == 'Yes':
                    pass
        else:
            print 'The date must be a string'
            return
        
        if place:
            self._data.update({date:bdata})
            self._process()
    
    def _process(self):
        ''' This processes new data added to the Rat object '''
        
        dates = self._to_process()
        
        processed = np.zeros(len(dates), dtype = self._session_records)
        
        trials = map(self._process_session, dates)
        
        for ii, packed in enumerate(zip(dates, trials)):
            processed[ii]['date'] = packed[0]
            processed[ii]['trials'] = packed[1]
            processed[ii]['n_trials'] = len(packed[1])
        
        #Calculate performances here
        performances = map(self._performance, processed)
        
        for ii, perfs in enumerate(performances):
            processed[ii]['performance'] = perfs
        
        self.sessions = np.concatenate((self.sessions, processed))
        self.sessions.sort(order='date')
    
    def _process_session(self, date):
        
        bdata = self._data[date]
        consts = bdata['CONSTS']
        
        records = [('outcome', 'i8'), ('response','i8'), ('stimulus', 'a20'), 
        ('block', 'a8'), ('hits', 'u1'), ('errors', 'u1'), ('correct side', 'i8'), 
        ('2PG port', 'i8'), ('PG port','i8'), ('FG port','i8'),
        ('2PG outcome', 'i8'), ('PG outcome','i8'), ('FG outcome','i8'),
        ('PG time','f8',2), ('RS time', 'f8'), ('FG time', 'f8'),
        ('C time', 'f8', 2), ('2PG response'), ('PG response','i8'), ('response','i8'),
        ('Trial length', 'f8'), ('block', 'i8')]
        
        n_trials = len(bdata['onsets'])
        
        stimuli = bdata['SOUNDS_INFO']['sound_name']
        stimuli = dict([ (ii, str(sound)) for ii,sound in enumerate(stimuli, 1)])
        
        # Unfortunately this is hard coded for the cued LLRR memory task
        # But it's the only task with blocks I'm working with right now
        blocks = dict({1:'cued', 2:'uncued'})
        
        trials = np.zeros(n_trials, dtype = records)
        
        trials_info = bdata['TRIALS_INFO'][:n_trials]
        
        trials['outcome'] = trials_info['OUTCOME']
        trials['hits'] = trials_info['OUTCOME'] == consts['HIT']
        trials['errors'] = trials_info['OUTCOME'] == consts['ERROR']
        trials['correct side'] = trials_info['CORRECT_SIDE']
        trials['stimulus'] = np.array([ stimuli[x] for x in trials_info['STIM_NUMBER']])
        
        # Will need to fix this...
        trials['block'] = np.array([tr[3:] for tr in trials['stimulus']])
        
        trials['response'][find(trials['hits'])] = \
            trials['correct side'][trials['hits']]
        incorrect_side = trials['correct side'][find(trials['errors'])]
        swap = {1:2, 2:1}
        swapped_sides = np.array([swap[t] for t in incorrect_side], dtype = 'uint8') 
        trials['response'][find(trials['errors'])] = swapped_sides
        
        trials['FG port'] = trials['correct side']
        trials['PG port'] = cat(np.zeros(1), trials['FG port'][:-1])
        trials['2PG port'] = cat(np.zeros(2), trials['FG port'][:-2])
        trials['FG outcome'] = trials['outcome']
        trials['PG outcome'] = cat(np.zeros(1), trials['FG outcome'][:-1])
        trials['2PG outcome'] = cat(np.zeros(2), trials['FG outcome'][:-2])
        trials['PG response'] = cat(np.zeros(1), trials['response'][:-1])
        trials['2PG response'] = cat(np.zeros(2), trials['response'][:-2])
        
        return trials
        
    def _to_process(self):
        
        data_dates = self._data.keys()
        process_dates = self.sessions['date']
        
        dates_to_process = [ date for date in data_dates if date not in process_dates ]
        
        return dates_to_process
        
    def _performance(self, session):
        
        records = session['trials']
        
        # Calculate first order performance
        all_perf = sum(records['hits']) / float(session['n_trials'])
        cued_perf = (sum(records['hits'] & (records['block'] == 'cued')) /
            float(sum(records['block'] == 'cued')))
        uncued_perf = (sum(records['hits'] & (records['block'] == 'uncued')) /
            float(sum(records['block'] == 'uncued')))
        
        # Calculate the number of trials to complete an uncued block
        cued = find(records['block'] == 'cued')
        block_len = np.diff(cued) - 1
        block_len = block_len[block_len != 0]
        
        perfs = {'all':all_perf, 'cued':cued_perf, 'uncued':uncued_perf, 
                        'blocks':block_len}
        
        return perfs

def batch(data_directory):
    
    import os
    import re
    
    filelist = os.listdir(data_directory)
    
    fileinfo = np.zeros(len(filelist), [('filename', 'a50'), ('rat', 'a7'),
        ('date', 'a9')])

    for ii, file in enumerate(filelist):
        reg=re.search('data_@TwoAltChoice_(\w+)_(\w+)_(\w+)_(\d+)(\D).mat', file)
        try:    
            fileinfo[ii]=(reg.group(0), reg.group(3), reg.group(4))
        except:
            pass
    
    fileinfo = fileinfo[fileinfo['filename'] !='']
    
    fileinfo.sort(order=['rat', 'date'])
    
    ratnames = np.unique(fileinfo['rat'])
    
    # Create rat objects for each rat in the batch
    rats = { name:Rat(name) for name in ratnames }
    
    for info in fileinfo:
        bdata = get_data("%s%s" % (data_directory, info['filename']))
        rats[info['rat']].update(info['date'], bdata)
    
    return rats
    
def get_data(filename):
    
    bload = bcontrol.Bcontrol_Loader(filename, mem_behavior = True, auto_validate = 0) 
    bdata = bload.load()
    bcontrol.process_for_saving(bdata)
    
    return bdata
    
def streaks(rats):
    
    """ Okay, I want to find consecutive hits during uncued blocks 
        
        So, what I'll do first is grab the 
        """
    
    d_list = []
    for rat in rats.itervalues():
        trials = rat.sessions['trials']
        uncued = [ find(trial['block'] == 'uncued') for trial in trials ]
        cued = [ find(trial['block'] == 'cued') for trial in trials ]
        
        un_blk_lens = [ np.diff(session)-1 for session in cued ]
        un_blk_lens = np.array([ sess[find(sess != 0)] for sess in uncued_lens ])
        
        un_hits = [ trial['hits'][uncued[ii]] for ii, trial in enumerate(trials) ]
        un_hits = [ find(hts == 1) for hts in un_hits ]
        
        # Grab the first block of uncued trials
        un_blk_lens
        
        
        d_list.append(diffs)
    
    return d_list
		
	
	
	
	
	
