#!/usr/bin/env python
'''
Quickly make the six-circle comparison plots for final ExHAIL proposal.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator as ML

import gmoutflow as gmo
import ExHail as ex

# Output directory:
outdir = './six_circle_plots/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# Parameters:
nOffset, nAzim = 10, 25
mhd = gmo.read_shell('./shell_r300_southward.dat')
ex.remap_shell(mhd, 1400)


def make_plot(azim, off, inc):
    '''
    Create a six-frame plot showing MHD results and ExHAIL reconstructions
    for 1 through 5 satellites.

    off, azim, and inc set the satellite offset, azimuth rotation, and
    inclination of orbit crossing point.
    '''

    # Create figure & decorate:
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=.03, bottom=.07, top=.96, right=.87, hspace=.04)
    fig.text(.45, .05, r'To Sun $\longrightarrow$', size=20,
             va='center', ha='center')

    # Create observation circle:
    out = gmo.add_flux_plot(mhd, target=fig, loc=231, zmax=1E12, colat_max=40)
    out[1].set_axis_off()
    out[1].set_title('"Observations" from MHD')

    # Add ExHAIL reconstructions:
    for nsat in range(1, 6):
        azim_all = [azim+x for x in np.linspace(-off, off, nsat)]
        const = ex.Constellation(azim_all, nsat*[inc], mhd)

        # Interpolated flux plot:
        ax = const.add_flux_obs(target=fig, loc=231+nsat,
                                contour=out[-1], colat_max=40.)
        const.add_orbit_lines(target=ax[1], ls='-', c='gray', lw=2.5)
        ax[1].set_title('{} Satellite{}'.format(
            nsat, 's'*bool(nsat-1)))

    # add colorbar:
    cb = fig.colorbar(out[-1], cax=fig.add_axes([.9, .15, .02, .7]),
                      orientation='vertical', extend='both', ticks=ML(2.5e11))
    cb.set_label(r'$Flux$ ($m^{-2}s^{-1}$)')
    return fig


for i, azim in enumerate(np.linspace(-180, 180, nAzim)):
    if i != 18:
        continue
    for j, offset in enumerate(np.linspace(1, 50., nOffset)):
        if j != 8:
            continue
        for k, inclin in enumerate([1, 5, 10, 15, 20, 25]):
            fig = make_plot(azim, offset, inclin)
            fig.savefig(outdir +
                        f'nsat_compare_{i:02d}_{j:02d}_{k:02d}.eps')
            plt.close('all')
