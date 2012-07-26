'''This is the module that will contain functions used in behavior analysis.
Author: Mat Leonard
Last modified: 6/8/2012
'''

import numpy as np
import scipy as sp
import scipy.io
import os
import re
import pickle as pkl
from matplotlib.mlab import find
from numpy import linalg
import matplotlib.pyplot as plt

class Rattatat:
    
    def get_perf(self, block = 'uncued'):
        ''' Returns a list of performances over sessions.
        
        Arguments:
        
        block: 'cued' or  'uncued'
        '''
        
        block = eval('self.' + block)
        perfs = [np.sum(self.hits[ii] & block[ii])/np.float(np.sum(block[ii]))
            for ii in range(len(self.hits))]
                
        return np.array(perfs)

    def hit_streaks(self):
        
        un_hits = [self.hits[ii] & self.uncued[ii] \
            for ii in np.arange(len(self.hits))]
        
        streaks = []
        ratio = []
        
        for sess in un_hits:
            prep = np.concatenate((sess.astype(int), np.zeros(1)))
            streak = (np.diff(find(prep == 0)) - 1)
            
            # Remove zeros
            streak = streak[find(streak != 0)]
            streaks.append(streak)
            
            ratio.append(np.sum(streak >= 6)/ np.float(np.sum(streak < 6)))
        
        self.streaks = streaks
        
        return ratio
    
    def strategies(self):
        
        streaks = self.hit_streaks()
        
        uncued = find([ len(s) > 1 for s in self.streaks ])
        
        sess_strats = [ self._fit_strats(ii) for ii in uncued ]
        
        block_match =  [ sess[1] for sess in sess_strats ]
        
        return block_match
    
    
    def _fit_strats(self,s_num):
        
        # We're going to want to fit models to the data, for each block
        
        big_data = (self.choice[s_num] - 1.5)*2
        big_errs = self.errs[s_num].astype(int)
        big_errs = -1*big_errs
        big_errs[find(big_errs == 0)] = 1
        inds = find(self.uncued[s_num])
        
        # Find the indices at the beginning of each uncued block
        firsts = [ inds[sum(self.uncued_lens[s_num][:ii])] 
            for ii in np.arange(len(self.uncued_lens[s_num]))]
        
        match = []
        blocks = []
        match_sum = []
        
        for jj in np.arange(len(firsts)):
            
            # Grab one uncued block
            trial_data = big_data[firsts[jj]:(firsts[jj]
                + self.uncued_lens[s_num][jj])]
            
            trial_errs = big_errs[firsts[jj]:(firsts[jj]
                + self.uncued_lens[s_num][jj])]
            
            #patch_len = len(trial_data)
                
            # Build the switch on errors model
            x_e = np.array([0]*len(trial_data))
            x_e[0] = trial_data[0]
            for ii in np.arange(len(x_e)-1):
                x_e[ii+1] = x_e[ii] * trial_errs[ii]
            
            # Build the LRLR model
            x_1 = np.array([0]*len(trial_data))
            x_1[0] = trial_data[0]
            for ii in np.arange(len(x_1)-1):
                x_1[ii+1] = -x_1[ii]
            
            # Build the LLRR model
            x_2 = np.array([0]*(len(trial_data)+1))
            x_2[1] = trial_data[0]
            x_2[0] = big_data[firsts[jj]-1]
            for ii in np.arange(1,len(x_2)-1):
                x_2[ii+1] = -x_2[ii-1] * trial_errs[ii-1]

            # This is an index of how well the models fit the data
            ind_e = np.dot(x_e, trial_data)/len(trial_data)
            ind_1 = np.dot(x_1, trial_data)/len(trial_data)
            ind_2 = np.dot(x_2[1:], trial_data)/len(trial_data)
            
            R = (ind_e, ind_1, ind_2)

            match.append(R)

            blocks.append(len(match)-1)
        
        # Find where the blocks are

        blocks = np.array(blocks) + np.ones(len(blocks))
        blocks = blocks.astype(int)
        if blocks[0] == 0:
            iterblocks = zip(np.concatenate((np.zeros(1).astype(int),
                blocks[:-1])),blocks)
        else:
            iterblocks = zip(blocks[:-1],blocks[1:])
        
        # For each block, average the match index
        for x,y in iterblocks:
            
            assert x != y, 'Got a problem here'
            avg = np.sum(match[x:y], axis=0)/(y-x)
            
            #assert not isnan(avg)
            
            match_sum.append(np.sum(match[x:y], axis=0)/(y-x))
  
        return match, match_sum, blocks 

def gettimes(data):

    hstry=data['saved_history'][0,0];
    y=hstry.ProtocolsSection_parsed_events.flatten();
    
    rtime = np.zeros(600);
    
    for ii in sp.arange(len(y)):
        if ii>=600:
            break
        z=y[ii][0]
        a=z.pokes[0,0];
        last_cpoke = a.C.max(axis=0).max()
        b=z.states[0,0];
        
        rtime[ii] = last_cpoke - b.play_stimulus[0,0];
                
    return rtime.tolist()

def build_rats(datadir = None):
    ''' Returns a list of rat objects containing information about rats
    from the data files in the datadir data directory.
    '''
    
    
    #Load information about the .mat files and sort by date.
    #datadir = '/home/mat/Documents/Behavior-data/Attention/5-3-2011/'
    #datadir = '/Users/Mattitude/Documents/Behavior-Data/Attention/5-3-2011/'
    
    
    #Get list of files in datadir
    filelist = os.listdir(datadir);
    
    #Then we need to extract the name of the rat and the date
    #Initialize structured array to store the identifying data
    fileinfo = np.zeros(len(filelist), [('file', 'a50'), ('rat', 'a7'),
                            ('date', 'a9')]);

    #Store filenames, rat names and dates
    for ii in np.arange(len(filelist)):
        reg=re.search('data_@TwoAltChoice_(\w+)_(\w+)_(\w+)_(\d+)(\D).mat',
                    filelist[ii]);
        try:    
            fileinfo[ii]=(reg.group(0), reg.group(3), reg.group(4));
        except:
            pass
    
    fileinfo = fileinfo[fileinfo['file']!=''];
    
    #Sort by rat name, then date
    fileinfo.sort(order=['rat', 'date']);

    #Load .mat file containing behavior data
    x=sp.io.loadmat(datadir+fileinfo['file'][-1], struct_as_record=False);
    temp = x['saved'][0,0];
    
    #First save column and trial numbers (hit, miss, left, right, etc)
    colms=temp.TwoAltChoice_Memory_TRIALS_INFO_COLS[0,0];
    consts=temp.TwoAltChoice_Memory_CONSTS[0,0];
    
    
    #check for trialdata, if not already there, load data!
    if 'trialdata' in os.listdir(datadir):
        
        x=file(datadir +'trialdata');
        saved_data = pkl.load(x);
        trialdata = saved_data[0]
        stims = saved_data[1]
        x.close();
    else:
        #Now let's iterate through each session and pull out the trial info
        trialdata=np.zeros(len(fileinfo), [('data', 'i2',
                    np.shape(temp.TwoAltChoice_Memory_TRIALS_INFO)), 
                    ('choice', 'f4'), ('mask', 'f4'),
                    ('delay', 'f4'), ('N', 'i4'),
                    ('rtime', 'f4', (600,))]);
        stims = [0]*len(fileinfo);
        for ii in np.arange(len(fileinfo)):
            x=sp.io.loadmat(datadir+fileinfo['file'][ii],
                        struct_as_record=False);
            temp=x['saved'][0,0];
            trialdata['data'][ii]=temp.TwoAltChoice_Memory_TRIALS_INFO[:600];
            trialdata['choice'][ii]=temp.TwoAltChoice_Memory_MaxChoiceTime;
            trialdata['mask'][ii]=temp.TwoAltChoice_Memory_Mask_Vol;
            trialdata['delay'][ii]=temp.TwoAltChoice_Memory_shortpoke_timeout;
            trialdata['N'][ii]=temp.ProtocolsSection_n_completed_trials;
            
            # Get sound info
            sound_info = temp.TwoAltChoice_Memory_SOUNDS_INFO[0,0];
            sounds = [ sound[0].encode() for sound in sound_info.sound_name[0] ]
            stims[ii] = dict(zip(sounds,np.arange(len(sounds))+1));
            
            #trialdata['rtime'][ii]=gettimes(x)[:600];
        
        saved_data = [trialdata, stims]
        
        x=file(datadir + 'trialdata','w');
        pkl.dump(saved_data,x);
        x.close();
            #turns out this is slow, can I do it faster? Nope, loading the
            #.mat files is just slow.  
    del x

    #Pull out rat names
    names = sorted(set(fileinfo['rat'])); 	#This gives a list of the rat names

    #This is the main structure the data will go in
    rats = list(np.zeros(len(names)));

    #Okay now, iterate through each rat and get the session data together
    for name in names:
        # Get the data that belongs to rat with 'name'
        ind = np.nonzero(fileinfo['rat']==name);
        
        #Filter out sessions here
        filt = np.nonzero(trialdata['N'][ind[0].tolist()]>=100);
        ind = ind[0][filt];
        
        ind = ind.tolist();
        
        rat = Rattatat();
        rat.name = name;
        rat.dates = fileinfo['date'][ind];
        rat.sess = list(np.zeros(len(ind)));
        #rat.rtime = list(np.zeros(len(ind)));
        #rat.choice = trialdata['choice'][ind];
        #rat.mask = trialdata['mask'][ind];
        rat.delay = trialdata['delay'][ind];
        rat.n_trials = trialdata['N'][ind];
        rat.sounds = [ stims[jj] for jj in ind ];
        
        for ii in ind:
            rat.sess[ind.index(ii)] = \
                trialdata['data'][ii][:trialdata['N'][ii]]
            #rat.rtime[ind.index(ii)] = trialdata['rtime'][ii];
        
        rats[names.index(name)] = rat;
    
    # Build hits and that sort of stuff
    for rat in rats:
    
        rat.hits = [ sess[:,colms.OUTCOME[0,0]-1] == consts.HIT[0,0] \
            for sess in rat.sess];
                
        rat.errs = [ sess[:,colms.OUTCOME[0,0]-1] == consts.ERROR[0,0] \
            for sess in rat.sess];
        
        stim_dict = dict()
        for sounds in rat.sounds:
            stim_dict.update(sounds)
        
        rat.stims = dict(stim_dict)
        
        for stim in rat.stims.keys():
            
            rat.stims[stim] = [ sess[:,colms.STIM_NUMBER[0,0]-1] == \
                    stim_dict[stim] for sess in rat.sess ];
                    
        # Turns out the block column in the trial data is broken.  Don't know
        # why it doesn't work.  So we're going to get the blocks based on the
        # stimuli played instead.
        
        rat.cued = [ rat.stims['le_cued'][ii] | rat.stims['ri_cued'][ii] \
            for ii in np.arange(len(rat.hits)) ];
            
        rat.uncued = [ rat.stims['le_uncued'][ii] | rat.stims['ri_uncued'][ii] \
            for ii in np.arange(len(rat.hits)) ];
            
        # What direction did the rat go 
        rat.choice = []
        for sess in rat.sess:
            side = sess[:,colms.CORRECT_SIDE[0,0]-1]
            outcome = sess[:,colms.OUTCOME[0,0]-1]
            choice = np.zeros(np.shape(side))
            
            # If the trial was a hit, then the choice was to the correct side
            choice[outcome==consts.HIT[0,0]] = side[outcome==consts.HIT[0,0]]
            
            # The trial was an error, then the choice was to the incorrect side
            choice[outcome==consts.ERROR[0,0]] = \
                ((side - 2)*(-1)+1)[outcome==consts.ERROR[0,0]]
            rat.choice.append(choice)
        
        # Uncued block lengths
        rat.uncued_lens = [];
        for sess in rat.cued:
            len_ind = (np.diff(find(sess))-1)
            uncued = len_ind[find(len_ind != 0)]
            
            rat.uncued_lens.append(uncued)
        
    return rats

def strats_perf(rats):
    plt.figure()
    for rat in rats:
        plt.plot([ np.mean(sess, axis=0)[2] 
            for sess in rat.strategies() if len(sess) != 0], label = rat.name)
    plt.legend()
    