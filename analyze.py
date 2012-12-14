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

def raster(unit, trialData, sort_by= 'PG in', range = None):
    
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
    if range == None:
        plt.xlim(srtdata['PG in'].mean()-2, srtdata['FG in'].mean()+2)
    else:
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
    
def trajectory(unit, trialData, range= (-10,3)):
    
    data = trialData
    
    # Define the plot
    def base_plot(unit, data, range):
        
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
            rates = ratehist(unit, traj_data[ii], range = range)
            plt.plot(rates.index, rates.T.mean())
            plt.title(titles[ii])
            plt.xlim(range)
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
            raster(unit, traj_data[ii], range = range)
            plt.title(titles[ii])
            
    cued = data[data['hits'] & (data['block']=='cued')]
    uncued = data[data['hits'] & (data['block']=='uncued')]
    
    # Pass the data to the plot
    base_plot(unit, cued, range)
    base_plot(unit, uncued, range)

def interval(unit, trialData, low, high):
    
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
    

def interval_scatter(units, locked, compare = 'PG', label = 'Avg rate'):
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

def slice(unit, trialData, sort_by = None, show = False):
    
    data = trialData
    
    if sort_by in trialData.columns:
        data = trialData.sort(columns=sort_by)
    
    rates = DataFrame(index=data.index, columns = range(6))
    
    for ind, row in data.iterrows():
        pg = (row['PG in'], row['PG out'])
        fg = (row['C out'], row['FG in'])
        cent = (row['C in'], row['C out'])
        delay = (row['PG out'], row['C in'])
    
        counts = [ np.histogram(row[unit.id], bins = 1, range=pg)[0] ]
        
        counts.append(np.histogram(row[unit.id], bins=3, range=delay)[0])
        counts.extend([np.histogram(row[unit.id], bins = 1, range=period)[0] 
                        for period in [cent, fg]])
        
        counts = np.concatenate(counts)
        diffs = [pg[1]-pg[0], (delay[1]-delay[0])/3.0, (delay[1]-delay[0])/3.0,
                (delay[1]-delay[0])/3.0, cent[1]-cent[0], fg[1]-fg[0], ]
        
        
        rates.ix[ind] = counts/diffs
    
    if show:
        plt.imshow(rates.astype(float), aspect='auto', interpolation = 'nearest',
            extent=[0,5,0,len(rates)])
    
    return rates

def ssi(rates_stims):
    ''' Calculates the Stimulus Selectivity Index.
    
    Parameters
    ----------
    rates_stims : pandas DataFrame
        A DataFrame with two columns, one column should be 'rates',
        the other column is 'stimulus', rows are trials.
    
    '''
    
    data = rates_stims
    stims = dict.fromkeys(np.unique(data['stimulus']))
    for stim in stims.iterkeys():
        just_stim = data[data['stimulus']==stim]
        rate = just_stim['rates'].mean()
        stims[stim] = rate
    
    avg_rates = stims.values()
    if (avg_rates[0]+avg_rates[1]):
        out = (avg_rates[0]-avg_rates[1])/(avg_rates[0]+avg_rates[1])
    else:
        out = 0
        
    
    return out

def ssi_scatter(timelock, iter = 100):
    
    from myutils import bootstrap
    from matplotlib.mlab import prctile, find
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    storeSSI = {'PG':[],'FG':[]}
    mem = {'PG response':'PG', 'response':'FG'}
    
    units = timelock.units
    
    # For each unit, compute the SSI for PG and FG
    for unit in units:
        
        data = timelock.get(unit)
        
        # For now, I only want to look at hit-hit trials
        select = (data['PG outcome']==consts['HIT']) & \
                (data['outcome']==consts['HIT'])
        trials = data[select]
        
        inter = interval(unit,trials,'PG out','onset')
        counts = inter[unit.id].map(len)
        rates = counts/(inter['onset']-inter['PG out'])
        for goal in mem.keys():
            input = DataFrame({'rates':rates, 'stimulus':trials[goal]})
            
            storeSSI[mem[goal]].append(bootstrap(input,ssi,iters=iter))
    
    meanSSI = dict.fromkeys(storeSSI.keys())
    intervSSI = dict.fromkeys(storeSSI.keys())
    
    for key, ssis in storeSSI.iteritems():
        # Calculate the means of the bootstrapped SSIs
        meanSSI[key] = [ np.mean(unitSSI) for unitSSI in ssis ]
        # Calculate the 95% confidence intervals of the boostrapped SSIs
        intervSSI[key] = [ prctile(unitSSI,p=(2.5,97.5)) for unitSSI in ssis ]
    
    # Now let's check for significance
    sig = dict.fromkeys(meanSSI.keys())
    def check_between(check, between):
        is_it = (between[0] <= check) & (between[1] >= check)
        return is_it
    for key, iSSIs in intervSSI.iteritems():
        sig[key] = np.array([ not check_between(0,issi) for issi in iSSIs ])
    
    not_sig = [ not (pg | fg) for pg,fg in zip(sig['PG'],sig['FG']) ]
    not_sig = np.array(not_sig)
    
    sig_colors = {'PG':'r','FG':'b'}
    xpnts = np.array(meanSSI['PG'])
    ypnts = np.array(meanSSI['FG'])
    xbars = np.abs(np.array(intervSSI['PG']).T - xpnts)
    ybars = np.abs(np.array(intervSSI['FG']).T - ypnts)
    
    # First, plot the not significant units
    ax.errorbar(xpnts[not_sig],ypnts[not_sig],
                yerr=ybars[:,not_sig],xerr=xbars[:,not_sig],
                fmt='o', color = 'grey')
    
    # Then plot things that are significant for PG and FG
    for key in sig.iterkeys():
        if sig[key].any():
            ax.errorbar(xpnts[sig[key]],ypnts[sig[key]],
                yerr=ybars[:,sig[key]],xerr=xbars[:,sig[key]],
                fmt='o', color = sig_colors[key])
    
    xs = ax.get_xlim()
    ys = ax.get_ylim()
    ax.plot(xs,[0,0],'-k')
    ax.plot([0,0],ys,'-k')
    ax.plot([-10,10],[-10,10],'--',color='grey')
    ax.set_xlabel('PG SSI')
    ax.set_ylabel('FG SSI')
    ax.set_xlim(xs)
    ax.set_ylim(ys)
    ax.set_aspect('equal')
    
    #fig.show()
    
    return sig, not_sig
