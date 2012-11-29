import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from itertools import izip
from bhv import constants
from sklearn.decomposition import PCA

consts = constants()

def ratehist(unit, trialData, bin_width=0.200, range = (-20,2)):
    ''' Output is a DataFrame, time as the index, columns are trials'''
    
    unitData = trialData[unit.id]
    nbins = np.diff(range)[0]/bin_width
    index = np.arange(range[0], range[1], bin_width)+bin_width/2.
    rateFrame = DataFrame(index = index)
    for ii, times in enumerate(unitData):
        count, x = np.histogram(times, bins = nbins, range = range)
        rate = count/bin_width
        rateFrame[ii] = rate
    return rateFrame

def raster(unit, trialData, sort_by= 'PG in', range = (-20,3)):
    
    uid = unit.id
    data = trialData
    
    srtdata = data.sort(sort_by, ascending = False)
    index = srtdata.index
    for trial, times in enumerate(srtdata[uid]):
        ind = index[trial]
        plt.scatter(times, np.ones(len(times))*trial, c = 'k', marker = '|', s = 5)
        plt.scatter(srtdata['PG in'][ind], trial, color = 'red', marker = 'o')
        plt.scatter(srtdata['FG in'][ind], trial, color = 'blue', marker = 'o')
    
    plt.plot([0,0], [0,len(data)], color = 'grey')
    plt.xlim(range)
    plt.ylim(0,len(data))
    plt.show() 

def basic_figs(unit, trialData, range=(-10,3)):
    
    data = trialData
    uid = unit.id
    times = ratehist(unit, data, range= range).T
    
    def base_plot(xs, rates, label):
        
        plt.figure()
            
        for rate, lab in izip(rates, label):
            plt.plot(xs, rate, label = lab)
        
        plt.plot([0,0], plt.ylim(), '-', color = 'grey')
        plt.xlabel('Time (s)')
        plt.ylabel('Activity (s)')
        plt.legend()
        
        
    # First plot all data
    base_plot(times.columns.values,[times.mean()], label = ['All trials'])
    
    # Plot PG left vs PG right
    pgleft = times[data['hits'] & (data['PG response']==consts['LEFT'])]
    pgright = times[data['hits'] & (data['PG response']==consts['RIGHT'])]
    rates = [pgleft.mean(), pgright.mean()]
    base_plot(times.columns.values, rates, label = ['PG left', 'PG right'])
    
    # Plot FG left vs FG right
    fgleft = times[data['hits'] & (data['response']==consts['LEFT'])]
    fgright = times[data['hits'] & (data['response']==consts['RIGHT'])]
    rates = [fgleft.mean(), fgright.mean()]
    base_plot(times.columns.values, rates, label = ['FG left', 'FG right'])
    
    # Plot cued vs uncued
    cued = times[data['hits'] & (data['block']=='cued')]
    uncued = times[data['hits'] & (data['block']=='uncued')]
    rates = [cued.mean(), uncued.mean()]
    base_plot(times.columns.values, rates, label = ['cued', 'uncued'])
    

def trajectory(unit, trialData, prange= (-10,3)):
    
    data = trialData
    
    def base_plot(unit, data, prange):
        
        ylims=[0]*4
        titles = ['right->left', 'left->left', 'left->right', 'right->right']
        
        
        traj = [('RIGHT', 'LEFT'), ('LEFT', 'LEFT'), ('LEFT', 'RIGHT'), 
                ('RIGHT', 'RIGHT')]
        
        traj_data = [ data[(data['PG response'] == consts[tr[0]])
                    & (data['response'] == consts[tr[1]])] for tr in traj]
        
        subplots = np.arange(221,225)
        
        # Making the rate histograms
        plt.figure()
        for ii, num in enumerate(subplots):
            plt.subplot(num)
            rates = ratehist(unit, traj_data[ii], range = prange)
            plt.plot(rates.index, rates.T.mean())
            plt.title(titles[ii])
            plt.xlim(prange)
            ylims[ii] = plt.ylim()[1]
        
        # Scaling all subplots to the same max y
        ymax = np.max(ylims)
        
        for num in subplots:
            plt.subplot(num)
            plt.ylim(0,ymax)
        
        # Now making the raster plots
        plt.figure()
        for ii, num in enumerate(subplots):
            plt.subplot(num)
            raster(unit, traj_data[ii], range = prange)
            plt.title(titles[ii])
            
    cued = data[data['hits'] & (data['block']=='cued')]
    uncued = data[data['hits'] & (data['block']=='uncued')]
    
    base_plot(unit, cued, prange)
    base_plot(unit, uncued, prange)

def epoch(unit, trialData, low, high):
    
    lows = ['PG in','PG out','C in','C out','L reward','R reward','onset']
    highs = ['PG out','C in','C out','L reward','R reward','onset','FG in']
    
    if low in lows:
        pass
    else:
        raise ValueError, '%s not a valid option for low' % low
        
    if high in highs:
        pass
    else:
        raise ValueError, '%s not a valid option for high' % high
    
    data = trialData
    times = data[[unit.id, low, high]].T
    for ii, series in times.iteritems():
        spiketimes = series.ix[0]
        lowtime = series.ix[1]
        hightime = series.ix[2]
        eptimes = spiketimes[(spiketimes>=lowtime) & (spiketimes<=hightime)]
        times[ii].ix[0] = eptimes
    
    return times.T
    

def epoch_scatter(units, locked, compare = 'PG', label = 'Avg rate'):
    ''' compare: valid options are 'PG', 'FG', 'block', 'repeats'
    '''
    
    events = [('PG in', 'PG out'), ('PG out', 'C in'), ('C in', 'C out'),
        ('C out', 'FG in')]
    
    comparisons = ['PG', 'FG', 'block', 'repeats']
    if compare in comparisons:
        pass
    else:
        raise ValueError, '%s not a valid option for compare' % compare
    
    for unit in units:
        
        data = locked[unit.session]
        
        if compare == 'PG':
            xdata = data[data['PG response'] == consts['LEFT']]
            ydata = data[data['PG response'] == consts['RIGHT']]
            xlabel = label + ', PG left'
            ylabel = label + ', PG right'
        elif compare == 'FG':    
            xdata = data[data['FG response'] == consts['LEFT']]
            ydata = data[data['FG response'] == consts['RIGHT']]
            xlabel = label + ', FG left'
            ylabel = label + ', FG right'
        elif compare == 'block':    
            xdata = data[data['block'] == 'cued']
            ydata = data[data['block'] == 'uncued']
            xlabel = label + ', cued'
            ylabel = label + ', uncued'
        elif compare == 'repeats':
            print 'Not implemented yet'
            return None
    

        x_epochs = [epoch(unit, xdata, ev[0], ev[1]) for ev in events]
        y_epochs = [epoch(unit, ydata, ev[0], ev[1]) for ev in events]
        
        x_boot = []
        y_boot = []
        
        for ep in x_epochs:
            
            
            
            rates = np.array([ len(spks)/ep[1][ii]  for data in ep ])
            x_mean.append(_bootstrap(rates, measure_func))
        
        for ep in y_epochs:
            
            rates = np.array([ len(spks)/ep[1][ii]  for ii, spks in enumerate(ep[0]) ])
            y_mean.append(_bootstrap(rates, measure_func))
        
        units.append((x_mean, y_mean))
        
    titles = ['In PG', 'PG to center', 'In center', 'Center to FG']
    
    for ii in range(len(events)):
        plt.figure()
        for unit in units:
            x_mean = unit[0]
            y_mean = unit[1]
            x, xconfs = x_mean[ii]
            y, yconfs = y_mean[ii]
            yerrlow = np.array([y - yconfs[0]])
            yerrhigh = np.array([yconfs[1] - y])
            xerrlow = np.array([x - xconfs[0]])
            xerrhigh = np.array([xconfs[1] - x])
            x = np.array([x])
            y = np.array([y])
            #1/0
            plt.errorbar(x, y, yerr = [yerrlow, yerrhigh], xerr = [xerrlow, xerrhigh],
                fmt='o', ecolor = 'grey', mfc = 'grey')
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(titles[ii])
        plt.plot([0,plt.xlim()[1]], [0,plt.ylim()[1]], '--k')
        #plt.xlim(0,plt.xlim()[1])
        #plt.ylim(0,plt.ylim()[1])
        plt.show()

def cut(unit, data, sort_by='PG in'):
    
    rates = DataFrame(index=data.index, columns = range(6))

    for ind, row in data.iterrows():
        pg = (row['PG in'], row['PG out'])
        fg = (row['C out'], row['FG in'])
        cent = (row['C in'], row['C out'])
        delay = (row['PG out'], row['C in'])
    
        counts = [np.histogram(row[unit.id], bins = 1, range=period)[0] 
            for period in [pg, fg,cent]]
        
        counts.append(np.histogram(row[unit.id], bins=3, range=delay)[0])
        counts = np.concatenate(counts)
        diffs = [pg[1]-pg[0], fg[1]-fg[0], cent[1]-cent[0], 
            (delay[1]-delay[0])/3.0, (delay[1]-delay[0])/3.0,
            (delay[1]-delay[0])/3.0]
        rates.ix[ind] = counts/diffs
    
    plt.imshow(rates.astype(float), aspect='auto', interpolation = 'nearest',
        extent=[0,5,0,len(rates)])
    
    return rates

def components(rates):
    prep = rates - rates.mean()
    pca = PCA(n_components = 2)
    out = pca.fit(prep).transform(prep)
    plt.figure()
    plt.scatter(out[:,0], out[:,1])
    
def fano(unit, data):
    
    spkcount = DataFrame(index=data.index, columns = range(6))
    
    for ind, row in data.iterrows():
        pg = (row['PG in'], row['PG out'])
        fg = (row['C out'], row['FG in'])
        cent = (row['C in'], row['C out'])
        delay = (row['PG out'], row['C in'])
    
        counts = [np.histogram(row[unit.id], bins = 1, range=period)[0] 
            for period in [pg, fg,cent]]
        
        counts.append(np.histogram(row[unit.id], bins=3, range=delay)[0])
        counts = np.concatenate(counts)
        
        spkcount.ix[ind] = counts
        fanofactor = spkcount.var()/spkcount.mean()
    return fanofactor.values
        
        
        