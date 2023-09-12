#!/usr/bin/env python
'''
Create some output files for the SWRI visualization guys.
'''

import numpy as np
from numpy import arctan2, arcsin, rad2deg

import matplotlib.pyplot as plt

import gmoutflow as gmo
import ExHail as ex

# Parameters to get good looking
nsat = 4
azim = -180
incl = 5
offset = np.linspace(1, 50., 10)[5]

# MHD:
if 'mhd' not in globals():
    mhd = gmo.read_shell('./shell_r300_southward.dat')
    ex.remap_shell(mhd, 1400)

# Set up each sat:
azim_all = [azim+x for x in np.linspace(-offset, offset, nsat)]

print('{} sats at {}incl and {}azim'.format(nsat, incl, azim))

# Get constellation:
const = ex.Constellation(azim_all, nsat*[incl], mhd)

# Quick plot:
fig = plt.figure(figsize=(10.36, 4.6))
fig.subplots_adjust(left=.03, right=.97)

# MHD flux:
f, a1, cnt = gmo.add_flux_plot(mhd, target=fig, loc=131, zmax=1E12,
                               colat_max=50)
a1.set_axis_off()

# Satellite path flux:
a2 = const.add_flux_lines(target=fig, loc=132, contour=cnt, colat_max=50.)
# Interpolated flux:
a3 = const.add_flux_obs(target=fig, loc=133, contour=cnt, colat_max=50.)
const.add_orbit_lines(target=a3[1], ls='-', c='gray')

# Create output files:
for i, sat in enumerate(const['sats']):
    out = open('demosat_{:02d}.txt'.format(i+1), 'w')

    # Calculate lat/lon
    r = np.sqrt(sat['x']**2+sat['z']**2+sat['y']**2)
    lat = rad2deg(arcsin(sat['z']/r))
    lon = rad2deg(arctan2(sat['y'], sat['x']))

    # Header:
    out.write('GSM X (Re)\tGSM Y (Re)\tGSM Z (Re)\t' +
              'Mag Lat (deg)\tMag Lon (deg)\tFlux (cm-3 s-1)\n')

    for i in range(sat['x'].size):
        out.write('{:.4F}\t{:.4F}\t{:.4F}\t{:8.3F}\t{:8.3F}\t{:.4E}\n'.format(
            sat['x'][i], sat['y'][i], sat['z'][i],
            lat[i], lon[i], sat['flux'][i]))

    out.close()
