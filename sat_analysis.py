#!/usr/bin/env python3
'''
Let's investigate how well or poorly we can reproduce the outflow pattern
given a certain number of spacecraft, an inclination angle of the node
of spacecraft convergence, and a range of spacecraft offsets and azimuths.
'''

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import gmoutflow as gmo
import ExHail as ex

# Set up argruments:
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('-i', '--incl', default=1, help="Set the inclination of" +
                    " the orbit crossing point.  Vaules should be between 1" +
                    " and 25 degrees (default=1)", type=int)
parser.add_argument('-n', '--nsats', default=4, type=int, help="Set the" +
                    " number of satellites (default=4)")
parser.add_argument('-o', '--offsets', default=10, help="Set the number of " +
                    " offset steps (default=10)", type=int)
parser.add_argument('-a', '--azims', default=25, type=int, help="Set the " +
                    "number of azimuthal steps (default=25)")
parser.add_argument('--minOff', default=1, type=float, help="Set the minimum" +
                    " azmuthal offset between the original azimuth and the " +
                    "outer satellite (1/2 total separation)")
parser.add_argument('--maxOff', default=50, type=float, help="Set the " +
                    "maximum azmuthal offset between the original azimuth " +
                    "and the outer satellite (1/2 total separation)")

args = parser.parse_args()

# Convenience vars from our parsed arguments:
nsat = args.nsats
incl = args.incl
nOffset, nAzim = args.offsets, args.azims

# Constants, paramters, etc.
mhd = gmo.read_shell('./shell_r300_southward.dat')
ex.remap_shell(mhd, 1400)

outdir = 'results_{:02d}sats_incl{:02d}'.format(nsat, incl)

# Make output directory.
if not os.path.isdir(outdir):
    os.mkdir(outdir)

out = open(outdir+'/results_{}sats.txt'.format(nsat), 'w')
out.write('ExHAIL fluence estimates for inclination=' +
          '{:02d}deg, nOffset={:03d}, nAzim={:03d}\n'.format(
              incl, nOffset, nAzim))
out.write('Separation\tAzimuth\tFlu_MHD  ' +
          '\tFlu_ExHail\tAbsErr\tRelErr\tRMSE\tCC\n')

for i, azim in enumerate(np.linspace(-180, 180, nAzim)):
    for j, offset in enumerate(np.linspace(args.minOff, args.maxOff, nOffset)):
        iPlot = i*nOffset+j

        azim_all = [azim+x for x in np.linspace(-offset, offset, nsat)]

        print('Working on #{}: satFirst={:5.2f}deg, satLast={:6.2f}deg'.format(
            iPlot, azim_all[0], azim_all[-1]))
        const = ex.Constellation(azim_all, nsat*[incl], mhd)
        f_mhd = const['flu_mhd']

        # Fluences:
        df = const['flu'] - f_mhd
        df_norm = df/f_mhd * 100.

        # Other metrics:
        rms = const.calc_rms()
        cc = const.calc_cc2d()

        # 2d correlation:
        # modshape = [const.attrs['nlat'], const.attrs['nlon']]
        # mod_flux = const['flux'].reshape(modshape)
        # corr = correlate2d(mhd['flux'], const['flux'], boundary='wrap' )
        # print corr

        # Quick plot:
        fig = plt.figure(figsize=(10.36, 4.6))
        fig.subplots_adjust(left=.03, right=.97)

        # MHD flux:
        f, a1, cnt = gmo.add_flux_plot(mhd, target=fig, loc=131, zmax=1E12,
                                       colat_max=50)
        a1.set_axis_off()

        # Satellite path flux:
        a2 = const.add_flux_lines(target=fig, loc=132, contour=cnt,
                                  colat_max=50.)
        # Interpolated flux:
        a3 = const.add_flux_obs(target=fig, loc=133, contour=cnt,
                                colat_max=50.)
        const.add_orbit_lines(target=a3[1], ls='-', c='gray')

        # Titles:
        a1.set_title('"Observations" from MHD\n{:10.2E} $ions/s$'.format(f_mhd))
        a2[1].set_title('ExHAIL Observations')
        a3[1].set_title('Reconstruction\n{:10.2E} $ions/s$'.format(const['flu']))

        fig.text(.5, .935, r'To Sun $\longrightarrow$', size=20,
                 va='center', ha='center')

        label = 'Total RMS Error:{:10.2E}$ions/s/cm^3$'.format(rms)
        fig.text(.5, .1, label, size=20, ha='center', va='center')

        # Save figure & close:
        f.savefig(outdir+'/fig_{:05d}.png'.format(iPlot))
        plt.close('all')

        df = const['flu'] - f_mhd
        df_norm = df/f_mhd * 100.
        out.write('{:6.2f}d\t{:7.2f}d\t{:13.5E}\t{:13.5E}\t{:13.5E}\t{:6.2f}%\t{:10.4E}\t{:7.5f}\n'.format(
            2*offset, azim, f_mhd, const['flu'], df, df_norm, rms, cc))
out.close()

# Sound bell when done.
sys.stdout.write('\a')
