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
            ('performance', object), ('trial_records', object)]
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
        
        trial_records = map(self._process_session, dates)
        
        for ii, packed in enumerate(zip(dates, trial_records)):
            processed[ii]['date'] = packed[0]
            processed[ii]['trial_records'] = packed[1]
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
        
        trial_records_dtypes = [('correct side', 'u1'), ('hits', 'u1'), 
            ('errors', 'u1'), ('response', 'u1'), ('stimulus', 'a20'),
            ('block', 'a20')]
        
        n_trials = len(bdata['onsets'])
        
        stimuli = bdata['SOUNDS_INFO']['sound_name']
        stimuli = dict([ (ii, str(sound)) for ii,sound in enumerate(stimuli, 1)])
        
        # Unfortunately this is hard coded for the cued LLRR memory task
        # But it's the only task with blocks I'm working with right now
        blocks = dict({1:'cued', 2:'uncued'})
        
        trial_records = np.zeros(n_trials, dtype = trial_records_dtypes)
        
        trials_info = bdata['TRIALS_INFO'][:n_trials]
        
        trial_records['hits'] = trials_info['OUTCOME'] == consts['HIT']
        trial_records['errors'] = trials_info['OUTCOME'] == consts['ERROR']
        
        trial_records['correct side'] = trials_info['CORRECT_SIDE']
        
        trial_records['stimulus'] = np.array([ stimuli[x] for x in trials_info['STIM_NUMBER']])
        
        # Will need to fix this...
        trial_records['block'] = np.array([tr[3:] for tr in trial_records['stimulus']])
        
        trial_records['response'][find(trial_records['hits'])] = \
            trial_records['correct side'][trial_records['hits']]
        incorrect_side = trial_records['correct side'][find(trial_records['errors'])]
        swap = {1:2, 2:1}
        swapped_sides = np.array([swap[t] for t in incorrect_side], dtype = 'uint8') 
        trial_records['response'][find(trial_records['errors'])] = swapped_sides
        
        return trial_records
        
    def _to_process(self):
        
        data_dates = self._data.keys()
        process_dates = self.sessions['date']
        
        dates_to_process = [ date for date in data_dates if date not in process_dates ]
        
        return dates_to_process
        
    def _performance(self, session):
        
        records = session['trial_records']
        
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
