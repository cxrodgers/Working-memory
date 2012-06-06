# This is a module for analyzing eletrophysiology recordings.

import scipy.special as spec
import scipy.optimize as opt
import numpy as np
import ns5
import scipy.signal as sig
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from sklearn import mixture
import pickle as pkl

# A class used for spike sorting.
class Spikesort(object):
    ''' A class used for spike sorting.  In general, you will load the data file
    with load_chans('path/to/data/file'), then just run sort().  From there you
    can cluster with different parameters and check ISIs and cross-correlations.
    
    sort() runs a few methods:
    get_spikes() : captures voltage spikes that cross a threshold
    get_tetrodes() : makes tetrode waveforms for each peak time found from get_spikes
    get_pca() : runs PCA on the tetrode waveforms to extract features
    get_clusters() : clusters the tetrode waveforms in the PCA space

    '''
    
    def __init__(self, filename=None):
        """ This method initiates the instance with:
            
            filename : path to the recording data file """
    
        self.filename = filename;
        
        # Create a dictionary for storing tetrode information
        self.tetrodes = { 'waveforms' : [] , 'peaks' : [], \
            'raw': [] }
            
        
    def load_chans(self, channels):
        """ This method loads and filters the specified channels from the 
        data file.  Filtering is done with a bandpass Butterworth filter, low
        cutoff = 300 Hz, high cutoff = 6 kHz.  Converts raw signal in bits
        to voltage, 4096.0 / 2^16 mV/bit.
        
        Arguments:
        
        channels : a list of channels you want to load
        """
        
        # Store number of channels loaded
        self.N = len(channels)
        
        # Initialize loader instance from ns5 module
        loader = ns5.Loader(self.filename);
    
        # Load header and file
        loader.load_header();
        loader.load_file();
        
        # Construct list of the raw data from each channel
        self.raw_chans = [ loader.get_channel_as_array(ii) for ii in channels ]     
        
        # Subtract the common mean from each channel
        cm = np.mean(self.raw_chans, axis = 0);
        cm_chans = self.raw_chans - cm;
        
        # Set parameters for bandpass filter
        filter_lo = 300 #Hz
        filter_hi = 6000 #Hz
        
        #Take the frequency, and divide by the Nyquist freq
        norm_lo= filter_lo/(30000.0/2)
        norm_hi= filter_hi/(30000.0/2)
        
        # Generate a 3-pole Butterworth filter
        b, a = sig.butter(3,[norm_lo,norm_hi], btype="bandpass");
        
        # Apply the filter to the raw data from each channel.  This removes
        # the LFP from the signal.  Also converts to voltage from bits,
        # the conversion factor is 4096.0 / 2^15 mV/bit.
        self.chans = [ sig.filtfilt(b,a,ch)*8192.0/2.**16 for ch in cm_chans ]
    
        # Also save the data, filtered, but without subtracting the common mean.
        self.filt_chans = [ sig.filtfilt(b,a,ch)*8192.0/2.**16 for ch in self.raw_chans ]
        
        if hasattr(self, 'clusters'):
            del self.clusters
        
    def sort(self,threshold = 5,K = 8, dims = 5, to_plot = True, auto_K = True):
        ''' This method combines everything up to clustering. '''
        self.get_spikes(threshold);
        self.get_tetrodes();
        
        try:
            self.get_pca(to_plot = False);
        except:
            print 'PCA broke!  Ruh roh!'
            print 'Number of spikes = ' + str(len(self.tetrodes['waveforms']))
            
        self.get_clusters(K, dims, 'full', to_plot, auto_K);
        
    def get_spikes(self,threshold):
        """ This function returns 40 sample sections of the data that cross 
        the threshold

        Arguments:
        
        threshold : this method catches pieces of the waveform that pass
            threshold*np.median(np.abs(data)/0.6745)
        """
        
        # Initialize these first
        self.peaks = [0]*self.N;
        self.spikes = [0]*self.N;
        
        # Find samples where the waveforms drop below threshold
        caught = [];
        
        for ch in self.chans:
            thresh = medthresh(ch,threshold);
            print 'Threshold is ' + str(thresh)
            caught.append( ch < thresh );
        
        caught = [ np.nonzero(x)[0] for x in caught ];
        
        for ii in np.arange(self.N):
        
            peak = 0;
            peaks = [];
            spikes = [];
            
            # Loop through each sample that crossed threshold
            for x in caught[ii]: 
                
                # Put a 15 sample buffer at the beginning and between peaks
                if x<15 or x<(peak+20):
                    pass 
                
                else:
                    # Grab the first crossing and 15 more samples
                    spike = self.chans[ii][x:(x+15)];
                    
                    # Find the minimum voltage, the peak
                    peak = x+np.argmin(spike);
                    
                    # Grab 30 samples around that peak
                    spike = self.chans[ii][(peak-10):(peak+20)];
                    
                    # Then, store the captured spike and peak (in samples)
                    spikes.append(spike);
                    peaks.append(peak);
    
            # Store the spikes and peaks for one channel.
            self.spikes[ii] = np.array(spikes);
            self.peaks[ii] = np.array(peaks);
            
        #self.data = [ {'waveform':self.chans[jj], 'channel':channels[ii]} \
        #    for ii in np.arange(len(channels))];
        
    def get_tetrodes(self):
        """ This function takes the spike peak times found after thresholding and
        returns waveforms which are a concatenation of the spikes from each
        individual channel waveform.
        """
        
        # Concatenate all the peak times from each channel
        cat_peaks = np.concatenate(self.peaks,axis=1);
        
        # Sort the peak times
        cat_peaks.sort();
        
        # Now we need to pick out separate peaks
        tetro_peaks = [];
        tetro_spikes = [];
        raw_tetro_spikes = [];
        peak = 0;
        
        for p in cat_peaks:
            # Skip this peak if it is within 20 samples of the previous peak
            if p - peak <= 20:
                pass
            else:
                # Grab 30 samples from each channel around previously found peak
                samps = [ self.chans[ii][(p-15):(p+15)] for ii in np.arange(4) ];
                
                # Find the channel with the peak
                ch = np.argmin(np.min(samps, axis=1));
                
                # Find the peak sample in that channel
                peak = (p - 15) + np.argmin(samps[ch]);
                
                # Look at the the only filtered data
                samps = [ self.filt_chans[ii][(peak-20):(peak+40)] for ii in np.arange(4) ];
                samps = np.array(samps)
                
                # If the waveform is pathological, don't store it
                if (samps > 200).any() or (samps < -500).any():
                    pass
                
                else:
                    # Store the raw waveform
                    cat_spikes = np.concatenate(samps);
                    raw_tetro_spikes.append(cat_spikes);
                
                    # Store the peak
                    tetro_peaks.append(peak);
                    
                    # Then grab 30 samples around that peak from each channel
                    samps = [ self.chans[ii][(peak-10):(peak+20)] for ii in np.arange(4) ];
                    
                    # Concatenate into one tetrode waveform
                    cat_spikes = np.concatenate(samps);
                    
                    # Store that waveform
                    tetro_spikes.append(cat_spikes);
                    
        tetro_peaks = np.array(tetro_peaks);
        tetro_spikes = np.array(tetro_spikes);
        raw_tetro_spikes = np.array(raw_tetro_spikes);
        
        # Create a dictionary storing the spike waveforms and the peak times
        self.tetrodes = { 'waveforms' : tetro_spikes, 'peaks' : tetro_peaks/30000.0, \
            'raw' : raw_tetro_spikes }
        
    def get_pca(self,dims=5, to_plot = True):
        """ This method performs PCA on the tetrode waveforms.  PCA can be
        considered as feature extraction.  It finds orthogonal basis vectors (components)
        in the waveform space that contain the most variance.
    
        Arguments:
        
        dims:  The number of PCA dimensions you want returned. default = 5
    
        """
        
        if hasattr(self, 'clusters'):
            # This will not run the first time through
            spikes = [ cluster['waveforms'] for cluster in self.clusters ];
            spikes = np.concatenate(spikes);
            
            peaks =  [ cluster['peaks'] for cluster in self.clusters ];
            peaks = np.concatenate(peaks);
            
            raw = [ cluster['raw'] for cluster in self.clusters ];
            raw = np.concatenate(raw);
            
            self.tetrodes['waveforms'] = spikes;
            self.tetrodes['peaks'] = peaks;
            self.tetrodes['raw'] = raw;
            
            del self.clusters
            
        else:
            spikes = self.tetrodes['waveforms']
            
        #Subtract the mean from the data before performing PCA
        spikes = spikes - np.mean(spikes, axis=0);
    
        # Perform PCA!  
        pca = PCA(n_components = dims)
        self.pca = pca.fit(spikes).transform(spikes)
        
        # This will generate a scatter plot of each waveform projected on
        # the first two PCA components.
        if to_plot:
            plt.figure();
            plt.plot(self.pca[:,0],self.pca[:,1],'ok',ms=1);
            plt.show()
    
    def get_clusters(self, K = 6, dims = 5, covariance = 'full', 
        to_plot=True, auto_K = False):
        """ This function clusters the spikes after decomposing the waveforms into 
        the PCA space.  Clustering is done using a Gaussian Mixture Model (GMM).
            
        Arguments:
        -----------------------------------------------------------------------
            
        K:  The number of Gaussians you want to fit to the data.  Each Gaussian
            can represent a class, cluster, or neuron.  Right now it is limited
            to 10 classes only because I haven't added more colors.
                
        dims:  This number of dimensions in which you want to fit the
            Gaussians.  Can't be more than the number of PCA components in 
            the PCA array.
                
        to_plot:  Plots the clusters in 2-dimensions, for the first three 
            PCA components.
                
        """
        
        # Create a vector of each waveform projected in PCA space
        if hasattr(self, 'clusters'):
            # This will not run the first time through
            pca = [ cluster['pca'] for cluster in self.clusters[1:] ]
            pca = np.concatenate(pca)
            waveforms = [ cluster['waveforms'] for cluster in self.clusters[1:] ]
            waveforms = np.concatenate(waveforms)
            raw = [ cluster['raw'] for cluster in self.clusters[1:] ]
            raw = np.concatenate(raw)
            peaks = [ cluster['peaks'] for cluster in self.clusters[1:] ]
            peaks = np.concatenate(peaks)
            noise = self.clusters[0]
        else:
            # This will run the first time through
            pca = self.pca[:,:dims]
            waveforms = self.tetrodes['waveforms']
            peaks = self.tetrodes['peaks']
            raw = self.tetrodes['raw']
        
        
        # This part will automatically find the best number of classes to use
        # in the model.  
        if auto_K == True:
            # Only test from 4 to 14 units
            for K in np.arange(4,14):
                # Instantiate the GMM 
                gmm = mixture.GMM(n_components = K, covariance_type = covariance, 
                    init_params='')
                
                # Fit the GMM to the data
                gmm.fit(pca)
                
                # If the model hasn't converged, keep training it
                while gmm.converged_ == False:
                    gmm.fit(pca)
                
                # We need to save the bic value from the previous model so we
                # can find the best model
                if 'prev_bic' not in locals():
                    prev_bic = gmm.bic(pca)
                elif gmm.bic(pca) > prev_bic:
                    # If the current number of classes (K) produces a worse model than
                    # the previous model, cluster with K-1 classes and stop
                    
                    K = K - 1
                    
                    gmm = mixture.GMM(n_components = K, covariance_type = covariance, 
                    init_params='')
                
                    # Fit the GMM to the data
                    gmm.fit(pca)
                    
                    # If the model hasn't converged, keep training it
                    while gmm.converged_ == False:
                        gmm.fit(pca)
                    
                    break
                else:
                    prev_bic = gmm.bic(pca)
                    
                
        else:
            # Instantiate the GMM 
            gmm = mixture.GMM(n_components = K, covariance_type = covariance, 
                init_params='')
            
            # Fit the GMM to the data
            gmm.fit(pca)
            
            # If the model hasn't converged, keep training it
            while gmm.converged_ == False:
                gmm.fit(pca)
        
        
        # Store the model after fitting
        self.model = gmm
        
        # Assign each data point to a Gaussian in the GMM
        assign = gmm.predict(pca);
        
        # Create a list of the waveform indices belonging to each cluster
        clusters = [ np.nonzero(assign == ii)[0] for ii in np.arange(K) ];
        
        # create a list of dictionaries, each dictionary is one cluster, separated
        # into 'waveforms' and 'peaks', which are the tetrode  voltage waveforms
        # and peak times respectively.
        self.clusters = [ { 'waveforms' : waveforms[cl], 'peaks' : peaks[cl], \
            'pca' : pca[cl], 'raw': raw[cl]} for cl in clusters ];
        
        # Let's add in a cluster just for noise and artifacts 
        self.clusters.insert(0, { 'waveforms' : 0, 'peaks' : 0, \
                'pca' : 0, 'raw': 0 })
        
        if 'noise' in locals():
            self.clusters[0] = noise
        
        if to_plot == True:
            
            self.plot_clust(np.arange(1,len(self.clusters)));
            

    def plot_clust(self, klusters = None):
        """ This function plots the clusters on the first three PCA components.
        
            Arguments:
            
            klusters : an array or list of the cluster numbers you want to plot
        """
        
        if klusters == None:
            klusters = np.arange(1,len(self.clusters));
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist();
        
        K = len(self.clusters);
        
        # Set the colors for all the plots based on the number of clusters
        self.colors = plt.cm.Paired(np.arange(0.0,1.0,1.0/K))
        
        plt.figure(1);
        plt.clf();
        plt.title('Tetrode waveforms plotted on PCA axes');
        
        for k in klusters:
        
            pca = self.clusters[k]['pca']
        
            # Plot clusters on PCA components 1 & 2
            plt.subplot(131);
            plt.scatter(pca[:,0], pca[:,1], marker = '.', linewidths=3, 
                facecolors = self.colors[k], edgecolors = 'none')
            plt.xlabel('PCA comp 1')
            plt.ylabel('PCA comp 2')
                
            # Plot clusters on PCA components 1 & 3
            plt.subplot(132);
            plt.scatter(pca[:,0], pca[:,2], marker = '.', linewidths=3, 
                facecolors = self.colors[k], edgecolors = 'none')
            plt.xlabel('PCA comp 1')
            plt.ylabel('PCA comp 3')   
                
            # Plot clusters on PCA components 2 & 3
            plt.subplot(133);
            plt.scatter(pca[:,1], pca[:,2], marker = '.', linewidths=3, 
                facecolors = self.colors[k], edgecolors = 'none')
            plt.xlabel('PCA comp 2')
            plt.ylabel('PCA comp 3')   
        
        #self.isi(klusters)
        
        # Adjust the figure margins and plot spacing
        plt.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.1);
        
        plt.show();
        
        self.mean_spikes(klusters)
        self.autocorr(klusters)
        
        
    def mean_spikes(self, klusters = None, to_plot = True):
        """ Calculates and returns the mean waveforms for each cluster.
        
            Arguments:
            
            to_plot : True to plot the mean waveforms & the standard deviation.
            
            klusters : a list of the cluster numbers you want to plot
        """
        
        if klusters == None:
            klusters = np.arange(len(self.clusters));
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist();
        
        self.means = [ np.mean(clst['raw'],axis=0) for clst in self.clusters ]
        
        # Check for nans and set them to 0
        nans = np.nonzero([ np.isnan(means).all() for means in self.means ])[0]
        for ii in nans:
            self.means[ii] = np.zeros(np.shape(self.means[ii]))
        
        
        if to_plot == True:
            
            plt.figure(2);
            plt.clf();
            
            n_rows = np.ceil(len(klusters)/3.0);
            
            wv_len = len(self.means[1])/self.N
        
            for k in klusters:
                
                waveforms = self.clusters[k]['raw'];
                
                # Draw 100 random waveforms from cluster
                # Check to be sure that cluster has more than 100 waveforms
                if len(waveforms) < 100:
                    rand_wfs = waveforms;
                else:
                    rand_wfs = waveforms[np.random.randint(0,len(waveforms),100)];
                
                plt.subplot(n_rows, 3, klusters.index(k) + 1);
                
                for ii in np.arange(self.N):
                    plt.plot(np.arange(5, wv_len + 5)+(wv_len+10)*ii,
                        rand_wfs[:,0+wv_len*ii:wv_len+wv_len*ii].T, 
                        color = self.colors[k], lw = 2, alpha = 0.2);
                    plt.plot(np.arange(5,wv_len+5)+(wv_len+10)*ii,
                        self.means[k][0+wv_len*ii:wv_len+wv_len*ii],
                        color = 'k', lw = 2);
                
                plt.title('cluster ' + str(k) + ', n = ' + str(len(waveforms)));
                            
            # Adjust the figure margins and plot spacing
            plt.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06, \
            hspace = 0.42);
            
            plt.show()
    
    def isi(self, klusters):
        """ Plots a histogram of inter-spike intervals (ISIs) for each cluster.  
        Spikes that are from real neurons should have ISIs that peak around 
        10-20 ms.  Spikes that are from noise should have ISIs that decay 
        exponentially.
        
        Arguments:
        
        klusters : and array or list of the cluster numbers you want to plot
        """
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist();
        
        plt.figure(3);
        plt.clf();
        n_rows = len(klusters)/3 + 1
    
        for k in klusters:
        
            peaks = self.clusters[k]['peaks'];
            
            # This finds the difference between each peak pair
            diff_peaks = [j-i for i, j in zip(peaks[:-1], peaks[1:])];
            
            # Set out the numbering for the subplots.  Probably need to fix this.
            plt.subplot(n_rows, 3, klusters.index(k) + 1);
            
            plt.hist(diff_peaks,bins = np.arange(0,0.1,0.001),color=self.colors[k])
            
            #plt.xlabel('ISI (s)');
            #plt.ylabel('Count');
            plt.xticks(alpha=1);
            plt.yticks(alpha=0.0);
            plt.title('cluster ' + str(k));
        
        # Adjust the figure margins and plot spacing
        plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06, \
            hspace = 0.42);
        
        plt.show();
        
    def autocorr(self, klusters = None, bin_width = 0.001, range = 0.02):
        """ This function calculates and plots the auto-correlation of clusters.
                
        Also plots the 95% probability that you'll see some number of events in
        a bin if the peak times are from a Poisson process, that is, if the events
        occur with some known average rate and indepdently of the time since the
        last event.
        
        In the future, I should also calculate the expected events for jittered
        times to control for average firing rate covariations.
        
        Arguments:
        
        klusters : an array or list of the cluster numbers you want to plot
        """
        
        if klusters == None:
            klusters = np.arange(len(self.clusters));
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist()
        
        plt.figure(4);
        
        plt.clf()
        
        n_rows = np.ceil(len(klusters)/3.0);
        if len(klusters) >= 3:
            n_cols = 3;
        else:
            n_cols = len(klusters);
        
        for k in klusters:
            
            peaks = self.clusters[k]['peaks'];
        
            # Get the bins and the events in each bin.
            correl = correlogram(peaks, bin_width = bin_width, 
                limit = range, auto = 1);
        
            # This is the average (expected) number of events in each bin
            #lam = np.ceil(sum(correl[0])/(len(correl[1])-1));
        
            # These functions calculate the cumulative distribution function (cdf)
            # for a Poisson process, shifted by 99% or 1% for finding the zero.
            # The cdf says, what is the probability that I will see k
            #ucdf = lambda k: 0.99 - spec.gammaincc(np.floor(k+1),lam);
            #lcdf = lambda k: spec.gammaincc(np.floor(k+1),lam) - 0.01;
            
            # Find the smallest number of events under 99% of the
            # cdf of the Poisson distribution
            #~ try:
                #~ u99 = np.floor(opt.brentq(ucdf,lam,lam+3*np.sqrt(lam)));
            #~ except:
                #~ u99 = lam + 2*np.sqrt(lam);
                #print 'expected events = ' + str(lam)
        
            # Find the largest number of events above 1% of the
            # cdf of the Poisson distribution
            #~ try:
                #~ u01 = np.ceil(opt.brentq(lcdf,a=0,b=lam));
            #~ except: 
                #~ u01 = 0.0;
                #print 'expected events = ' + str(lam)
        
            # Set out the numbering for the subplots. 
            plt.subplot(n_rows, n_cols, klusters.index(k)  + 1);
            
            # Plot the correlogram
            plt.bar(correl[1][:-1]*1000,correl[0], width = bin_width*1000, 
                color = self.colors[k], edgecolor = 'none');
        
            # Plot the 99% Poisson probability
            #plt.plot(correl[1],u99*np.ones(len(correl[1])), '--', color = 'gray', lw = 2);
        
            # Plot the 1% Poisson probability
            #plt.plot(correl[1],u01*np.ones(len(correl[1])), '--', color = 'gray', lw = 2);
        
            # If the peak times are drawn from a Poisson distribution, then events in each
            # bin has 98% probability of falling between these two lines.  If you see bins
            # that are either above or below these lines, they can be considered significant.
            
            plt.yticks(alpha=0.0);
            plt.title('cluster ' + str(k));
            
            #plt.title('Cluster ' + str(k+1));
                
        # Now actually draw the plot
        plt.subplots_adjust(left=0.04, right=0.97, top=0.94, bottom=0.06, \
            hspace = 0.42);
        plt.show()
        
    def correls(self, klusters = None, doodads = None):
        """ This function calculates and plots the cross-correlation of clusters.
                
        Also plots the 99% probability that you'll see some number of events in
        a bin if the peak times are from a Poisson process, that is, if the events
        occur with some known average rate and independently of the time since the
        last event.
        
        In the future, I should also calculate the expected events for jittered
        times to control for average firing rate covariations.
        
        Arguments:
        
        klusters : an array or list of the cluster numbers you want to plot
        """
        
        if klusters == None:
            klusters = np.arange(len(self.clusters));
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist()
        
        plt.figure(5);
        
        plt.clf();
        
        # Set the number of rows and columns to plot
        n_rows = len(klusters);
        n_cols = len(klusters);
        
        # Get the clusters used in the correlograms
        clusts = [self.clusters[k]['peaks'] for k in klusters];
        
        # Build a list used to iterate through all the correlograms
        l_rows = np.arange(n_rows+1)
        l_cols = np.arange(n_cols)
        ind = [(x, y) for x in l_rows for y in l_cols if y >= x]
        
        
        for ii, jj in ind:
            
            if ii == jj:
                auto = True;
            else:
                auto = False;
            
            # Get the bins and the events in each bin.
            correl = correlogram(clusts[ii], clusts[jj], auto = auto);
            
            if doodads == 'fancy':
                # This is the average (expected) number of events in each bin
                lam = sum(correl[0])/(len(correl[1])-1);
            
                # These functions calculate the cumulative distribution function (cdf)
                # for a Poisson process, shifted by 99% or 1% for finding the zero.
                # The cdf says, what is the probability that I will see x or y
                ucdf = lambda x: 0.99 - spec.gammaincc(np.floor(x+1),lam);
                lcdf = lambda y: spec.gammaincc(np.floor(y+1),lam) - 0.01;
                
                # Find the smallest number of events under 99% of the
                # cdf of the Poisson distribution
                try:
                    u99 = np.floor(opt.brentq(ucdf,lam,lam+3*np.sqrt(lam)));
                except:
                    u99 = lam + 2*np.sqrt(lam);
                    #print 'expected events = ' + str(lam)
            
                # Find the largest number of events above 1% of the
                # cdf of the Poisson distribution
                try:
                    u01 = np.ceil(opt.brentq(lcdf,a=0,b=lam));
                except: 
                    u01 = 0.0;
                   #print 'expected events = ' + str(lam)
                
                # Now that we have the Poisson probabilities, let's check for
                # covariations in the mean firing rate.  What we'll do is randomly 
                # jitter the peaks times to erase correlations while preserving
                # the average firing rate.
                
                j_correl = jitter(clusts[ii], clusts[jj], 5);
            
            
            # This part below here does all the plotting.
        
            k = ind.index((ii, jj));
            
            # Set out the numbering for the subplots. 
            plt.subplot(n_rows, n_cols, ii*n_cols + jj + 1);
            
            l = klusters[ii]
            
            if auto:
                # Plot the auto-correlogram
                plt.bar(correl[1][:-1],correl[0],width = 0.001, color = self.colors[l], 
                    edgecolor = 'none');
            else:
                # Plot the cross-correlogram
                plt.bar(correl[1][:-1],correl[0],width = 0.001, color = 'k');
                
                if doodads == 'fancy':
                    # Plot the jittered histogram
                    plt.plot(correl[1][:-1]+0.0005,j_correl, '--', color = 'r', lw = 2);
            
            if doodads == 'fancy':
                # Plot the 99% Poisson probability
                plt.plot(correl[1],u99*np.ones(len(correl[1])), '--', color = 'gray', lw = 2);
            
                # Plot the 1% Poisson probability
                plt.plot(correl[1],u01*np.ones(len(correl[1])), '--', color = 'gray', lw = 2);
            
            
            
            # If the peak times are drawn from a Poisson distribution, then events in each
            # bin has 99% probability of falling between these two lines.  If you see bins
            # that are either above or below these lines, they can be considered significant.
            
            plt.xticks(np.arange(-0.02,0.03, 0.01), ('-20', '-10', '0', '10', '20') ,alpha=1 );
            plt.yticks(alpha=0.0);
            
            #plt.title('Cluster ' + str(k+1));
                
        # Adjust the figure margins and plot spacing
        plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06, \
            hspace = 0.42);
        # Now actually draw the plot
        plt.show()
        
    def combine(self, klusters):
        ''' This method combines multiple clusters into one cluster.  Use this
        if the clustering algorithm assigns spikes from the same neuron to
        different clusters.
        
        Arguments:
        
        klusters : an array or list of the clusters you want to combine.
        '''
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist();
        
        # Sort the clusters to take out so the go from largest to smallest.
        # Doing this because the cluster list changes shape as you take out
        # clusters.
        klusters.sort(reverse=True);
        
        keys = self.clusters[0].keys();
        
        comb = dict().fromkeys(keys);
        
        # For each cluster, remove the desired cluster and add it to the
        # concatenated cluster
        for k in klusters:
            x = self.clusters.pop(k);
            for key in x.iterkeys():
                if comb[key] == None:
                    comb[key] = x[key];
                else:
                    comb[key]=np.concatenate((comb[key],x[key]));
    
        # Add concatenated dictionary back to clustered data
        self.clusters.append(comb);
            
        # And replot
        self.plot_clust(np.arange(len(self.clusters)));
        #self.autocorr(np.arange(len(self.cls)));
        #self.mean_spikes(np.arange(len(self.cls)));
        
    def split(self, kluster):
        ''' Splits a cluster into two clusters.  It takes the data in a cluster,
        runs PCA on it again, then clusters it again.
        '''
        
        cluster = self.clusters[kluster]
        
        # Get the spike waveforms and subtract the mean
        spikes = cluster['waveforms']
        spikes = spikes - np.mean(spikes, axis = 0)
        
        # Do some PCA on the spike waveforms
        pca = PCA(n_components = 5)
        pca_spikes = pca.fit(spikes).transform(spikes)
        
        # Now cluster!
        gmm = mixture.GMM(n_components = 2)
        
        # Fit the GMM to the data
        gmm.fit(pca_spikes);
        
        # If the model hasn't converged, keep training it
        while gmm.converged_ == False:
            gmm.fit(pca_spikes, init_params='');
        
        # Assign each data point to a Gaussian in the GMM
        assign = gmm.predict(pca_spikes);
        
        # Create a list of the waveform indices belonging to each cluster
        clusters = [ np.nonzero(assign == ii)[0] for ii in np.arange(2) ];
        
        new_clusters = [ { 'waveforms' : cluster['waveforms'][cl], 
            'peaks' : cluster['peaks'][cl], 
            'pca' : cluster['pca'][cl], 'raw': cluster['raw'][cl] } \
            for cl in clusters ];
        
        # Now remove the cluster we are splitting
        self.clusters.pop(kluster)
        
        # And add the two new clusters
        self.clusters.extend(new_clusters)
        
        self.plot_clust()
        
    def remove(self, klusters):
        ''' Adds a cluster to the noise cluster.
        
        Arguments:
        
        klusters : an array or list of clusters you want removed.
        '''
        
        if type(klusters) == np.ndarray:
            klusters = klusters.tolist();
            
        # Sort the clusters to take out so they go from largest to smallest.
        # Doing this because the cluster list changes shape as you take out
        # clusters.
        klusters.sort(reverse=True);
    
        # Remove the desired clusters
        for k in klusters:
            self.clusters[0].update(self.clusters.pop(k))
        
        self.plot_clust(np.arange(1, len(self.clusters)));
        
    def save_clusters(self, save_as):
        ''' Saves the tetrode waveforms and time stamps to a file for later
        analysis.  Load the file using pickle.
        
        Parameters:
        
        save_as : a string for the file name.
        '''
        
        fil = open(save_as + '.dat', 'w');
        
        pkl.dump(self.clusters, fil);
        
        fil.close();
    
    def load_clusters(self, filename):
        ''' This is used to load already clustered data so you can look at it
        or manipulate it if you want.
        '''
        
        if not hasattr(self, 'clusters'):
            self.clusters = []
        
        # Load the cluster data
        fil = open(filename,'r')
        loaded = pkl.load(fil)
        fil.close()
        
        # Then add the data to the sorter
        self.clusters.extend(loaded)
        
        self.N = len(self.clusters)
    
    def plot_drift(self, klusters):
        ''' This method plots the drift of a cluster over time.'''
        
        plt.figure(6)
        
        plt.clf()
        
        for k in klusters:
            plt.scatter(self.clusters[k]['peaks'],
                self.clusters[k]['pca'][:,0], marker = '.', linewidths = 3,
                edgecolor = 'none', facecolor = self.colors[k])
        
        plt.show
    
    def outliers(self):
        ''' This method is going to detect and remove outliers from each cluster.
        
        Outlier removal is done following the method in Hill D.N., et al., 2011
        
        '''     
        
        # Find the outliers for each cluster except the noise cluster        
        for ii, clst in enumerate(self.clusters[1:]):
            
            mean = self.means[ii]
            cov = np.matrix(np.cov(clst['raw'].T))
            
            diff = np.matrix(clst['raw'] - mean)
            invcov = np.matrix(np.linalg.inv(cov))
            
            # Calculate the chi^2 values for each data point
            chi2 = np.array([ (vec*invcov*vec.T).A[0,0] for vec in diff ])
            
            # Find outliers and inliers
            outliers = np.nonzero(chi2 < 1/len(diff))[0]
            inliers = np.nonzero(chi2 > 1/len(diff))[0]
            
            for key, values in clst.iteritems():
                # Save outliers to noise cluster
                np.concatenate((self.clusters[0][key], values[outliers]))
                
                # Save only the inliers
                clst[key] = values[inliers]
        
        self.plot_clust(np.arange(1,len(self.clusters)))



def jitter(peaks1, peaks2, jitter):
    
    jit1 = 0.001*np.random.randint(-jitter, jitter+1, (10,len(peaks1)));
    tiled1 = np.tile(peaks1,(10,1));
    jit_peaks1 = tiled1 + jit1;
    
    jit2 = 0.001*np.random.randint(-jitter, jitter+1, (10,len(peaks2)));
    tiled2 = np.tile(peaks2,(10,1));
    jit_peaks2 = peaks2 + jit2;
    
    a = [ correlogram(x, y, auto = 0)[0] for x in jit_peaks1 for y in jit_peaks2 ];
    
    return np.mean(a, axis=0)

def correlogram(t1, t2=None, bin_width=.001, limit=.02, auto=False):
    
    """Return crosscorrelogram of two spike trains.
    
    Essentially, this algorithm subtracts each spike time in `t1` 
    from all of `t2` and bins the results with numpy.histogram, though
    several tweaks were made for efficiency.
    
    Arguments
    ---------
        t1 : first spiketrain, raw spike times in seconds.
        t2 : second spiketrain, raw spike times in seconds.
        bin_width : width of each bar in histogram in sec
        limit : positive and negative extent of histogram, in seconds
        auto : if True, then returns autocorrelogram of `t1` and in
            this case `t2` can be None.
    
    Returns
    -------
        (count, bins) : a tuple containing the bin edges (in seconds) and the
        count of spikes in each bin.

        `bins` is relative to `t1`. That is, if `t1` leads `t2`, then
        `count` will peak in a positive time bin.
    """
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if auto: t2 = t1

    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)
    bins = np.linspace(-limit, limit, num=(2 * limit/bin_width + 1))

    # This is the old way to calculate bin edges. I think it is more
    # sensitive to numerical error. The new way may slightly change the
    # way that spikes near the bin edges are assigned.
    #bins = np.arange(-limit, limit + bin_width, bin_width)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to numpy.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] -= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins

def medthresh(data,threshold):
    """ A function that calculates the threshold based off the median value of
    the data.
    
    Arguments:
    
    data : your data
    threshold : the threshold multiplier
    """
    
    return -threshold*np.median(np.abs(data)/0.6745)

def tetrode(num):
    """
    Gives you a dictionary listing the channel numbers for each tetrode.
    I got tired of always having to look this up.
    """
    
    tetrodes = {1:[16,18,17,20],2:[19,22,21,24],3:[23,26,25,28],\
        4:[27,30,29,32]};
    
    return tetrodes[num]

def trace_plot(data, ylim):
    
    clf()
    plt.subplots_adjust(left=0.15);
    
    plt.subplot(411)
    plt.plot(np.arange(0,30000*180)/30000., data[0][:30000*3*60])
    plt.xticks(alpha = 0)
    plt.yticks(fontsize = 'large')
    plt.ylabel('Voltage (mV)', size = 'x-large')
    plt.ylim(ylim)
    
    plt.subplot(412)
    plt.plot(np.arange(0,30000*180)/30000.,data[1][:30000*3*60])
    plt.yticks(alpha = 0)
    plt.xticks(alpha = 0)
    plt.ylim(ylim)
    
    plt.subplot(413)
    plt.plot(np.arange(0,30000*180)/30000.,data[2][:30000*3*60])
    plt.xticks(alpha = 0)
    plt.yticks(alpha = 0)
    plt.ylim(ylim)
    
    plt.subplot(414)
    plt.plot(np.arange(0,30000*180)/30000., data[3][:30000*3*60])
    plt.xlabel('Time (s)', size = 'x-large')
    plt.xticks(fontsize = 'large')
    plt.yticks(alpha = 0)
    plt.ylim(ylim)