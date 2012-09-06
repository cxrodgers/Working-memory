''' This script syncs the behavior and neural data.  You'll run this
after spike sorting, before timelocking.  Cheers!
'''

import bcontrol 
import DataSession 
import AudioTools 
import ns5
import numpy as np
import pickle as pkl
     
# Need to create this class to work with the syncing algorithm
class Onsets(object):
    def __init__(self,onsets):
      self.audio_onsets = onsets
    
behave_file = '/media/hippocampus/WM_data/CWM019/data_@TwoAltChoice_Memory_Mat_CWM019_120509a.mat' 
neural_file = '/media/hippocampus/NDAQ/datafile_ML_CWM019_120509_001.ns5'
save_as = 'CWM019_120509'

loader = ns5.Loader(neural_file)
loader.load_header()
loader.load_file()
   
audio = [ loader.get_analog_channel_as_array(n) for n in [7,8] ] 
audio = np.array(audio)*4096./2**16
    
onsets_obj = AudioTools.OnsetDetector(audio, verbose = True, 
    minimum_threshhold=-20)
onsets_obj.execute()
n_onsets = Onsets(onsets_obj.detected_onsets)

    
bcload = bcontrol.Bcontrol_Loader(filename = behave_file, mem_behavior= True, auto_validate = 0) 
bcdata = bcload.load()
b_onsets = Onsets(bcdata['onsets'])
   
syncer = DataSession.BehavingSyncer()
syncer.sync(b_onsets,n_onsets, force_run = 1)
   
with open(save_as + '.bhv','w') as f:
	bcontrol.process_for_saving(bcdata)
	pkl.dump(bcdata,fil)

with open(save_as + 'ons','w') as f:
	pkl.dump(n_onsets.audio_onsets,f)
     
with open(save_as+ '.syn','w') as f:
	pkl.dump(syncer,f)
