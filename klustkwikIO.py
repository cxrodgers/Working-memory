# Iterate through all neurons in OE db and dump spiketimes
# We will use the KlustaKwik compatible Feature File format
# Ref: http://klusters.sourceforge.net/UserManual/data-files.html
# Begin format specification (lightly modified from above):
"""
The Feature File

Generic file name: base.fet.n

Format: ASCII, integer values

The feature file lists for each spike the PCA coefficients for each
electrode, followed by the timestamp of the spike (more features can
be inserted between the PCA coefficients and the timestamp). 
The first line contains the number of dimensions. 
Assuming N1 spikes (spike1...spikeN1), N2 electrodes (e1...eN2) and
N3 coefficients (c1...cN3), this file looks like:

nbDimensions
c1_e1_spike1   c2_e1_spike1  ... cN3_e1_spike1   c1_e2_spike1  ... cN3_eN2_spike1   timestamp_spike1
c1_e1_spike2   c2_e1_spike2  ... cN3_e1_spike2   c1_e2_spike2  ... cN3_eN2_spike2   timestamp_spike2
...
c1_e1_spikeN1  c2_e1_spikeN1 ... cN3_e1_spikeN1  c1_e2_spikeN1 ... cN3_eN2_spikeN1  timestamp_spikeN1

The timestamp is expressed in multiples of the sampling interval. For
instance, for a 20kHz recording (50 microsecond sampling interval), a
timestamp of 200 corresponds to 200x0.000050s=0.01s from the beginning
of the recording session.

Notice that the last line must end with a newline or carriage return. 
"""
import cPickle
import numpy as np
import os.path
import shutil

class UniqueError(Exception):
    pass

def unique_or_error(a):
    u = np.unique(np.asarray(a))
    if len(u) == 0:
        raise UniqueError("no values found")
    if len(u) > 1:
        raise UniqueError("%d values found, should be one" % len(u))
    else:
        return u[0]


data_dir = '/home/mat/Dropbox/Working-memory/CWM019/120508/'
filename = 'clusters_CWM019_120508_tet2.dat'
f_samp = 30e3
output_dir = data_dir
basename = os.path.splitext(filename)[0]
group = 1 # tetrode 1

# load data
dorun = False
try:
    data
except NameError:
    dorun = True
if dorun:
    with file(os.path.join(data_dir, filename)) as fi:
        data = cPickle.load(fi)
n_features = unique_or_error([d['pca'].shape[1] for d in data])
n_clusters = len(data)


# filenames
fetfilename = os.path.join(output_dir, basename + '.fet.%d' % group)
clufilename = os.path.join(output_dir, basename + '.clu.%d' % group)
spkfilename = os.path.join(output_dir, basename + '.spk.%d' % group)
#~ if os.path.exists(fetfilename):
    #~ shutil.copyfile(fetfilename, fetfilename + '~')
#~ if os.path.exists(clufilename):
    #~ shutil.copyfile(clufilename, clufilename + '~')

# write fetfile
with file(fetfilename, 'w') as fetfile:
    # Write an extra feature for the time
    fetfile.write('%d\n' % (n_features+1))
    
    # Write one cluster at a time
    for d in data:
        to_write = np.hstack([d['pca'], 
            np.rint(d['peaks'][:, None] * f_samp).astype(np.int)])    
        fmt = ['%f'] * n_features + ['%d']
        np.savetxt(fetfile, to_write, fmt=fmt)

# write clufile
with file(clufilename, 'w') as clufile:
    clufile.write('%d\n' % n_clusters)
    for n, d in enumerate(data):
        np.savetxt(clufile, n * 
            np.ones(len(d['peaks']), dtype=np.int), fmt='%d')

# write spkfile
with file(spkfilename, 'w') as spkfile:
    
    clst_wvs = [0]*len(data)
    for ii, clst in enumerate(data):
        # This gets each waveform into the format needed for the KlustaKwik
        # spike file format
        cl_spks = np.concatenate( [ np.reshape( np.reshape( wvform, \
            (4, len(wvform)/4)), len(wvform), order ='F') \
            for wvform in clst['waveforms'] ] )
        
        clst_wvs[ii] = (cl_spks/(8192.0/2.**16)).astype(np.int16)
        
    spks = np.concatenate(clst_wvs)
    
    spks.tofile(spkfile)


