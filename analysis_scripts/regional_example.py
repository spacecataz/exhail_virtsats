#!/usr/bin/env python
'''
Let's make an example "hot-spot" observation for Phase 1 ExHAIL.
This script is written to run in IPython's interactive mode.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.mlab import bivariate_normal

import gmoutflow as gmo
import ExHail as ex

# Set constants for hotspot size:
size = 300 / 6371.0  # width in km.
xspot, yspot = .39, .24  # center in RE.
amp = 7.5E11  # amplitude of spot.

# Orbit conditions to see hotspot:
nsat = 4
azim = -180+46.
incl = 5.0
offset = 15

# Stuff for plotting:
zmax = 1E12
nlevs = 101
levels = np.linspace(-1.*zmax, zmax, nlevs)

# Load and alter data if not done already:
if 'mhd' not in globals():
    # Start by loading MHD results:
    mhd = gmo.read_shell('./shell_r300_southward.dat')
    mhd['flux'] *= .2
    ex.remap_shell(mhd, 1400)

    # Add a "hot spot" on the dayside.  Normalize to flux values.
    spot = bivariate_normal(mhd['x'], mhd['y'], size, size, xspot, yspot)
    spot *= amp/spot.max()

    # Show addition of hotspot:
    fig = plt.figure()
    a1 = fig.add_subplot(131)
    a2 = fig.add_subplot(132)
    a3 = fig.add_subplot(133)
    a1.tricontourf(mhd['x'], mhd['y'], mhd['flux'], levels, cmap='seismic')
    a2.tricontourf(mhd['x'], mhd['y'], spot,        levels, cmap='seismic')

    mhd['flux'] = mhd['flux'] + spot
    a3.tricontourf(mhd['x'], mhd['y'], mhd['flux'], levels, cmap='seismic')

# Now create constellation:
azim_all = [azim+x for x in np.linspace(-offset, offset, nsat)]
const = ex.Constellation(azim_all, nsat*[incl], mhd, npoints=1601)

# Typical plot:
# Quick plot:
fig = plt.figure(figsize=(10., 10))
# fig.subplots_adjust(left=.03, right=.97)

# MHD flux:
f, a1, cnt = gmo.add_flux_plot(mhd, target=fig, loc=221, zmax=1E12,
                               colat_max=50)
a1.set_axis_off()
const.add_orbit_lines(target=a1, ls='-', c='gray')

# Satellite path flux:
a2 = const.add_flux_lines(target=fig, loc=222, contour=cnt, colat_max=50.)

for a in (a1, a2[1]):
    a.set_xlim([0, .8])
    a.set_ylim([0, .8])

# Titles:
a1.set_title('Regional Outflow from MHD')
a2[1].set_title('ExHAIL Observations')

a3 = fig.add_subplot(212)
for i, sat in enumerate(const['sats']):
    a3.plot(sat['flux'], label='ExHAIL {:d}'.format(i+1), lw=2.5)
a3.set_xlim([485, 545])
a3.set_ylim([-4e11, 6e11])
# a3.set_xticklabels('')
a3.set_xlabel('Time $\\rightarrow$', size=20)
a3.set_ylabel('Flux', size=20)
a3.legend(loc='best')
fig.tight_layout()


# Better plot:
fig = plt.figure(figsize=[8.5, 3.6])
fig.subplots_adjust(top=.93, left=0, bottom=.17, right=.98)

a1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
a2 = plt.subplot2grid((1, 5), (0, 2), colspan=3)

# MHD flux:
f, a1, cnt = gmo.add_flux_plot(mhd, target=a1, zmax=1E12, colat_max=50)
a1.set_axis_off()
const.add_orbit_lines(target=a1, ls='-', c='gray')
a1.set_xlim([0, .8])
a1.set_ylim([-.0001, .8])
a1.set_title('')

# Label satellites:
x = [.428, .555, .646, .70]
y = [.650, .600, .494, .36]
r = [67, 57, 46, 36]
for i, c in enumerate(['b', 'g', 'r', 'c']):
    # plt.plot(x, y, '+', color='k')
    a1.text(x[i], y[i], 'ExHAIL {}'.format(i+1), rotation=r[i], color=c,
            va='center', ha='center', zorder=100, size=12, weight='bold')
    # bbox={'fc':'w','ec':'w','alpha':.5})

# Observations:
colors = []
for i, sat in enumerate(const['sats']):
    a2.plot(sat['flux'], label='ExHAIL {:d}'.format(i+1), lw=2.5)

# Set up good axes labels of latitude:
lat = 180./np.pi*np.arctan2(sat['z'], np.sqrt(sat['x']**2+sat['y']**2))


def lat_formatter(x, pos):
    '''
    Set the latitude tick.
    '''
    return '{:.0f}$^{{\\circ}}$'.format(lat[int(x)])


a2.xaxis.set_major_formatter(FuncFormatter(lat_formatter))

# Finish customizing axes:
a2.set_xlim((450.7803751345516, 530.30038014944842))  # [454, 577] )
a2.set_ylim((-114824489924.33597, 722253973962.12561))  # [0, 7e11])
# a2.set_xticklabels('')
a2.set_yticklabels('')
a2.set_xlabel('Latitude', size=20)
a2.set_ylabel('Flux $\\longrightarrow$', size=20)
a2.legend(loc='best')
