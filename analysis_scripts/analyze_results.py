#!/usr/bin/env python

'''
Read the ascii output files for all of the four-sat-analysis runs and
plot the error as a function of tilt, separation, etc.
'''

import numpy as np

# Plotting parameters:
props = {'whiskerprops': {'linewidth': 1.5, 'linestyle': '-'},
         'capprops': {'linewidth': 1.5}, 'whis': 1.5,
         'boxprops': {'linewidth': 2}, 'medianprops': {'linewidth': 2},
         'showfliers': False}


def read_results(filename):
    '''
    Load and parse a results text file.
    '''
    import re

    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()

    # Parse header:
    match = re.findall('=(\d+)', lines.pop(0))
    incl, nSep, nAzim = float(match[0]), int(match[1]), int(match[2])
    lines.pop(0)  # Get rid of last header line.

    data = {'incl': incl, 'nSep': nSep, 'nAzim': nAzim, 'fart': 'a'}
    rawE = np.zeros(len(lines))
    rawS = np.zeros(len(lines))
    rawA = np.zeros(len(lines))
    rawR = np.zeros(len(lines))
    rawC = np.zeros(len(lines))

    for i, l in enumerate(lines):
        # l.replace('d', '')
        # l.replace('%', '')
        parts = l.split()
        rawS[i] = parts[0][:-1]
        rawA[i] = parts[1][:-1]
        rawE[i] = parts[5][:-1]
        rawR[i] = parts[6]
        rawC[i] = parts[7]

    data['rms'] = rawR.reshape((nAzim, nSep))
    data['error'] = rawE.reshape((nAzim, nSep))
    data['cc2d'] = rawC.reshape((nAzim, nSep))
    data['sep'] = rawS[:nSep]
    data['azim'] = rawA[::nSep]
    return data


def get_incl(f):
    '''
    From a file name, extract the inclination.
    '''
    import re

    a = re.search('incl(\d+)', f)
    return float(a.groups()[0])


def sort_all_results(nsats=4):
    from glob import glob
    import cPickle as cp

    # wordnum = {4:'four', 3:'three', 5:'five', 6:'six'}

    # List of all results files:
    files = glob('./results_{:02d}sats_incl*/results*.txt'.format(nsats))
    files.sort()

    # Number of files:
    nIncl = len(files)
    incl = []

    # Open first file, create container of appropriate size.
    f = files.pop(0)

    incl.append(get_incl(f))
    data = read_results(f)
    error = np.zeros((data['nAzim'], data['nSep'], nIncl))
    rms = np.zeros((data['nAzim'], data['nSep'], nIncl))
    cc2d = np.zeros((data['nAzim'], data['nSep'], nIncl))
    error[:, :, 0] = data['error']
    rms[:, :, 0] = data['rms']
    cc2d[:, :, 0] = data['cc2d']

    # Fill array with rest of data.
    for i, f in enumerate(files):
        # print f
        incl.append(get_incl(f))
        data = read_results(f)
        error[:, :, i+1] = data['error']
        cc2d[:, :, i+1] = data['cc2d']
        rms[:, :, i+1] = data['rms']

    # Add more info and create a data structure:
    incl = np.array(incl)
    alldat = {'err': error, 'sep': data['sep'], 'azim': data['azim'],
              'incl': incl, 'rms': rms, 'cc2d': cc2d}

    # Save data to file.
    out = open('./sats{}_error.cpk'.format(nsats), 'w')
    cp.dump(alldat, out)
    out.close()

    return alldat


if __name__ == '__main__':
    '''
    Open/parse all results files, sort into a numpy array, perform
    analysis.
    '''

    import os
    import pickle as cp

    import matplotlib.pyplot as plt

    import spacepy.plot as splot
    splot.style()

    # props={'whiskerprops':{'lw':1.5, 'ls':'-'}, 'capprops':{'lw':1.5},
    #        'boxprops':{'lw':2}, 'medianprops':{'lw':2}, 'showfliers':True}

    # Plotting parameters:
    props = {'whiskerprops': {'linewidth': 1.5, 'linestyle': '-'},
             'capprops': {'linewidth': 1.5}, 'whis': 1.5,
             'boxprops': {'linewidth': 2}, 'medianprops': {'linewidth': 2},
             'showfliers': False}

    # Load data:
    dats = {}
    for inc_read in [5, 7, 10, 12, 15, 17, 19]:
        for nsat in range(1, 7):
            dir_now = 'results_{:02d}sats_incl{:02d}'.format(nsat, inc_read)
            if os.path.exists('error_{:01d}sat.cpk'.format(nsat)):
                infile = open('error_{:01d}sat.cpk'.format(nsat))
                dats[nsat] = cp.load(infile)
                infile.close()
            elif os.path.exists(dir_now):
                dats[nsat] = sort_all_results(nsat)
            else:
                print("No results for {} sats (looking for {})".format(
                    nsat, dir_now))
    # This is helpful.
    nsats = dats.keys()
    nsats.sort()

    # #### ##### #####
    # #### PLOTS #####
    # #### ##### #####

    # ##### As a function of number of sats:
    # Collect relevant data:
    cc, rm, n = [], [], []  # empty containers.
    for i in dats:
        n.append(i)
        # Here, we're limiting the number of configurations included:
        # the indexing is [azims, separation, inclinations]
        # cc.append( dats[i]['cc2d'][:,5:,1:-2].flatten() )
        # rm.append( dats[i]['rms' ][:,5:,1:-2].flatten() )
        cc.append(dats[i]['cc2d'][3:9, -3:-1, :-1].flatten())
        rm.append(dats[i]['rms'][3:9, -3:-1, :-1].flatten())

    # Print some info to screen:
    cclast = 0
    for i, c in enumerate(cc):
        ccnow = np.median(c)
        frac = np.count_nonzero(c > .7)/float(c.size) * 100.
        print("{} sats: cc2d = {:.3f}({:+.03f}); {:.1f}% above .7".format(
            i+1, ccnow, ccnow-cclast, frac))
        cclast = ccnow

    #  Box plot representation
    #  Correlation coeff:
    # fig = plt.figure(figsize=(6.2,5.75))#figsize=(12.4,5.75))
    # ax  = fig.add_subplot(111)
    # ax.boxplot(cc, positions = n, widths=.75, **props)
    # ax.set_xlabel('Number of Satellites', size=18)
    # ax.set_ylabel('Correlation', size=18)
    # # RMS Error:
    # #ax  = fig.add_subplot(122)
    # #ax.boxplot(rm, positions = n, widths=.75, **props)
    # #ax.set_xlabel('Number of Satellites', size=18)
    # #ax.set_ylabel('RMS Error', size=18)
    # fig.suptitle('Reconstruction Quality vs. # of Spacecraft', size=20)
    # fig.tight_layout()
    # fig.subplots_adjust(top=.9)

    # ##### Correlation as function of mission time (i.e., separation).
    t1, t2, t3 = [], [], []
    azim1 = 3
    azim2 = 9
    for i in nsats:
        v1 = dats[i]['cc2d'][azim1:azim2, 0:3, :-1].flatten()
        v2 = dats[i]['cc2d'][azim1:azim2, 5:8, :-1].flatten()
        v3 = dats[i]['cc2d'][azim1:azim2, -3:, :-1].flatten()
        t1.append(np.median(v1))
        t2.append(np.median(v2))
        t3.append(np.median(v3))
        print(v1.size, v2.size, v3.size)

    fig = plt.figure(figsize=[7.05,  6.625])
    ax = fig.add_subplot(111)
    ax.plot(nsats, t1, 'bs-', ms=10, label='July-Oct. 2023')
    ax.plot(nsats, t2, 'yd-', ms=10, label='Nov. 2023 - Feb. 2024')
    ax.plot(nsats, t3, 'ro-', ms=10, label='March-June 2024')
    ax.set_title('Reconstruction Quality vs. # of Spacecraft', size=20)
    ax.set_xlabel('Number of Satellites', size=18)
    ax.set_ylabel('2D Correlation', size=18)
    ax.legend(loc='best')
    ax.set_xlim([.8, 6.2])
    ax.set_ylim((0.44635244831961118, 0.86027757884140654))  # [.507, .824] )
    fig.tight_layout()

    # # As a function of both azim and number of sats.
    # n_azim = dats[1]['azim'].size
    # azims = np.zeros( n_azim )
    # ccs = np.zeros( (n_azim, len(dats)) )
    # rms = np.zeros( (n_azim, len(dats)) )
    # for n in nsats:
    #     for m in range(n_azim):
    #         azims[m] = dats[1]['azim'][m]
    #         ccs[m, n-1] = np.median(dats[n]['cc2d'][m,:,:])
    #         rms[m, n-1] = np.median(dats[n]['rms' ][m,:,:])
    #
    # # 2D correlation:
    # fig = plt.figure()
    # ax  = fig.add_subplot(111)
    # img = ax.imshow(ccs)
    # ax.set_aspect('auto')
    #
    # ax.set_xlabel('Number of Satellites', size=20)
    # ax.set_ylabel('Mag. Lon. of Crossing Point', size=20)
    #
    # yticks = ['']+['{:+3.0f}'.format(az)+'$^{\\circ}$' for az in azims ]
    # xticks = ['']+['{:1d}'.format(n+1) for n in range(len(dats))]
    # ax.set_yticklabels(yticks)
    # ax.set_xticklabels(xticks)
    #
    # cbar = plt.colorbar(img)
    # cbar.set_label('Median Correlation', size=20)
    # cbar.set_ticks(MultipleLocator(.05))

    # # As a function of both inclination and number of sats.
    # n_incl = dats[1]['incl'].size
    # incls = np.zeros( n_incl )
    # ccs = np.zeros( (n_incl, len(dats)) )
    # rms = np.zeros( (n_incl, len(dats)) )
    # for n in nsats:
    #     for m in range(n_incl):
    #         incls[m] = 90-dats[1]['incl'][m]
    #         if incls[m] == 89: incls[m]+=1
    #         ccs[m, n-1] = np.median(dats[n]['cc2d'][:,:,m])
    #         rms[m, n-1] = np.median(dats[n]['rms' ][:,:,m])
    #
    # # 2D correlation:
    # fig = plt.figure()
    # ax  = fig.add_subplot(111)
    # img = ax.imshow(ccs)
    # ax.set_aspect('auto')
    #
    # ax.set_xlabel('Number of Satellites', size=20)
    # ax.set_ylabel('Mag. Lat. of Crossing Point', size=20)
    #
    # yticks = ['']+['{:2.0f}'.format(inc)+'$^{\\circ}$' for inc in incls ]
    # xticks = ['']+['{:1d}'.format(n+1) for n in range(len(dats))]
    # ax.set_yticklabels(yticks)
    # ax.set_xticklabels(xticks)
    #
    # cbar = plt.colorbar(img)
    # cbar.set_label('Median Correlation', size=20)
    # cbar.set_ticks(MultipleLocator(.05))
    #
    # # 2D RMS
    # fig = plt.figure()
    # ax  = fig.add_subplot(111)
    # img = ax.imshow(rms, cmap='plasma_r')
    # ax.set_aspect('auto')
    #
    # ax.set_xlabel('Number of Satellites', size=20)
    # ax.set_ylabel('Mag. Lat. of Crossing Point', size=20)
    #
    # yticks = ['']+['{:2.0f}'.format(inc)+'$^{\\circ}$' for inc in incls ]
    # xticks = ['']+['{:1d}'.format(n+1) for n in range(len(dats))]
    # ax.set_yticklabels(yticks)
    # ax.set_xticklabels(xticks)
    #
    # cbar = plt.colorbar(img)
    # cbar.set_label('Median RMS Error', size=20)
    # #cbar.set_ticks(MultipleLocator(.05))
    #
    #
    # #### AS A FUNCTION OF SEPARATION:
    # n_sep = dats[1]['sep'].size
    # seps  = dats[1]['sep'][-1:0:-2]
    # ccs = np.zeros( (n_sep, len(dats)) )
    # rms = np.zeros( (n_sep, len(dats)) )
    #
    # for n in nsats:
    #     for m in range(n_sep):
    #         ccs[m, n-1] = np.median(dats[n]['cc2d'][:,n_sep-1-m,:])
    #         rms[m, n-1] = np.median(dats[n]['rms' ][:,n_sep-1-m,:])
    #
    #
    # #### 2D correlation:
    # fig = plt.figure(figsize=[ 6.2,  7.575 ])
    # ax  = fig.add_subplot(111)
    # img = ax.imshow(ccs)
    #
    # ax.set_xlabel('Number of Satellites', size=20)
    # ax.set_ylabel('Max. Orbit Plane Separation', size=20)
    #
    # yticks = ['']+['{:3.0f}'.format(s)+'$^{\\circ}$' for s in seps ]
    # xticks = ['']+['{:1d}'.format(n+1) for n in range(len(dats))]
    # ax.set_yticklabels(yticks)
    # ax.set_xticklabels(xticks)
    # ax.set_aspect('auto')
    #
    # cbar = plt.colorbar(img)
    # cbar.set_label('Median Correlation', size=20)
    # cbar.set_ticks(MultipleLocator(.025))
    #
    # fig.tight_layout()
    #
    # ### 1D correlation:
    # #fig = plt.figure()
    # #ax = fig.add_subplot(111)
    # #ax.plot(dats[4]['sep'],ccs[::-1,3])
    # #ax.hlines(.7, 0, 100, linestyles='dashed', linewidths=2, colors='k')
    # #ax.set_title('4 Sat Constellation')
    # #ax.set_ylabel('2D Correlation with Reality')
    # #ax.set_xlabel('Orbit Plane Separation (degrees)')
    #
    # #### 2D RMS
    # fig = plt.figure(figsize=[ 6.2,  7.575 ])
    # ax  = fig.add_subplot(111)
    # img = ax.imshow(rms, cmap='plasma_r')
    #
    # ax.set_xlabel('Number of Satellites', size=20)
    # ax.set_ylabel('Max. Orbit Plane Separation', size=20)
    #
    # #yticks = ['']+['{:2.0f}'.format(inc)+'$^{\\circ}$' for inc in incls ]
    # xticks = ['']+['{:1d}'.format(n+1) for n in range(len(dats))]
    # ax.set_yticklabels(yticks)
    # ax.set_xticklabels(xticks)
    # ax.set_aspect('auto')
    #
    # cbar = plt.colorbar(img)
    # cbar.set_label('Median RMS Error', size=20)
    # cbar.set_ticks(MultipleLocator(.1E11))
    #
    # fig.tight_layout()
