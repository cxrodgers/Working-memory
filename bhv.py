''' Behavior data analysis '''

import bcontrol
import numpy as np
from matplotlib.mlab import find
from pandas import *

class Rat(object):
    
    def __init__(self, name):
        
        if not isinstance(name, basestring):
            raise TypeError, 'name should be a string'
        else:
            self.name = name
        self._data = {}
        self._session_records = [('n_trials', 'u2'), ('date', 'a6'), 
            ('performance', object), ('trials', object)]
        self.sessions = np.zeros(0, dtype = self._session_records)
    
    def update(self, date, bdata):
        
        if isinstance(date, basestring):
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
            print 'The date must be a string of form YYMMDD'
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
        
        # Set up the DataFrame
        records = ['outcome', 'response', 'stimulus', 'block', 'hits', 'errors', 
        'correct', '2PG port', 'PG port', 'FG port', '2PG outcome',
        'PG outcome', 'FG outcome','2PG response', 'PG response',
        'PG in', 'PG out', 'FG in','C in', 'C out', 'L reward',
        'R reward', 'onset']
        n_trials = len(bdata['onsets'])
        
        #  Create the DataFrame
        trials = DataFrame(index= range(n_trials), columns = records)
       
        trials_info = bdata['TRIALS_INFO'][:n_trials]
        
        trials['outcome'] = trials_info['OUTCOME']
        trials['hits'] = trials_info['OUTCOME'] == consts['HIT']
        trials['errors'] = trials_info['OUTCOME'] == consts['ERROR']
        trials['correct'] = trials_info['CORRECT_SIDE']
        
        stimuli = bdata['SOUNDS_INFO']['sound_name']
        stimuli = dict([ (ii, str(sound)) for ii,sound in enumerate(stimuli, 1)])
        trials['stimulus'] = np.array([ stimuli[x] for x in trials_info['STIM_NUMBER']])
        
        # Will need to fix this...
        trials['block'] = np.array([tr[3:] for tr in trials['stimulus']])
        
        trials['response'][find(trials['hits'])] = \
            trials['correct'][trials['hits']]
        incorrect_side = trials['correct'][find(trials['errors'])]
        swap = {1:2, 2:1}
        swapped_sides = np.array([swap[t] for t in incorrect_side], dtype = 'uint8') 
        trials['response'][find(trials['errors'])] = swapped_sides
        
        trials['FG port'] = trials['correct']
        trials['PG port'] = trials['FG port'].shift(1)
        trials['2PG port'] = trials['FG port'].shift(2)
        trials['FG outcome'] = trials['outcome']
        trials['PG outcome'] = trials['FG outcome'].shift(1)
        trials['2PG outcome'] = trials['FG outcome'].shift(2)
        trials['PG response'] =trials['response'].shift(1)
        trials['2PG response'] = trials['response'].shift(2)
        
        # Now for timing information
        
        trials['onset'] = bdata['onsets']
        
        from itertools import izip
        # Since I'll need the times from the current trial and the previous trial...
        iterpeh = izip(bdata['peh'], bdata['peh'][1:])
        trial = 1
        # I'm making a decision to skip the first trial since it is worthless
        for previous, current in iterpeh:
            
            trials['PG in'][trial] = np.nanmax(previous['states']['choosing_side'])
            trials['PG out'][trial] = np.nanmin(current['states']['state_0'])
            trials['C in'][trial]= np.nanmin(current['states']['hold_center'])
            
            trials['C out'][trial] = np.nanmin(current['states']['choosing_side'])
            
            trials['FG in'][trial] = np.nanmax(current['states']['choosing_side'])
           
            try:
                trials['L reward'][trial] = np.nanmin(current['states']['left_reward'])
            except ValueError:
                pass
            try:
                trials['R reward'][trial] = np.nanmin(current['states']['right_reward'])
            except ValueError:
                pass
        
            trial += 1
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

def constants():
    consts = {'CHOICE_TIME_UP': 3,
                    'CONGRUENT': 1,
                    'CURRENT_TRIAL': 6,
                    'ERROR': 1,
                    'FUTURE_TRIAL': 4,
                    'GO': 1,
                    'HIT': 2,
                    'INCONGRUENT': -1,
                    'LEFT': 1,
                    'NOGO': 2,
                    'NONCONGRUENT': -1,
                    'NONRANDOM': 2,
                    'RANDOM': 1,
                    'RIGHT': 2,
                    'SHORTPOKE': 5,
                    'SHORT_CPOKE': 5,
                    'TWOAC': 3,
                    'UNKNOWN_OUTCOME': 4,
                    'WRONG_PORT': 7}
    
    return consts


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
    