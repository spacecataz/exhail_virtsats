#!/usr/bin/env python

'''
A module for creating ExHail mission planning analysis.

Notes:

- Currently, this only operates for the northern hemisphere.
'''

import numpy as np
from spacepy.pybats import PbData
from spacepy.plot import set_target
import matplotlib.pyplot as plt


def cc2d(f, t):
    '''
    2D correlation coefficient for two matrices of the same size and shape.
    Based on "Fast Normalized Cross-Correlation" by J. P. Lewis; adapted
    so that no shifting takes place.  Doesn't get more simple than this.
    '''
    corr = ((f-f.mean()) * (t-t.mean())).sum()
    variance = np.sqrt(((f-f.mean())**2).sum() * ((t-t.mean())**2).sum())

    return corr/variance


def fprint(string_in):
    '''
    Print to screen and flush buffer.  Useful for debugging when the system
    is laggy or likely to lock.
    '''
    from sys import stdout

    print(string_in)
    stdout.flush()


def circle(radius, zenith, azimuth, npoints=401):
    '''
    Given a circle centered at the origin of a given *radius*,
    *zenith* angle (in degrees) of its highest point off the Z-axis, and
    an *azimuth* angle off of the positive x-axis, calculate and return
    the X, Y, and Z coordinates of *npoints* points evenly distributed about
    that circle.
    '''

    from numpy import sin, cos, pi

    # Convert angles to radians:
    phi, theta = pi*float(azimuth+90)/180., pi*float(zenith+90)/180.

    # Calculate perpendicular unit vector:
    uhat = np.array([-sin(phi), cos(phi), 0])
    # Calculate normal-cross-perpendicular:
    nxu = np.array([cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)])

    # Generate coordinates using parameterized equation:
    t = np.linspace(0, 2*pi, npoints)
    xyz = np.zeros([3, npoints])
    for j in range(3):
        xyz[j, :] = radius*cos(t)*uhat[j] + radius*sin(t)*nxu[j]

    return xyz[0, :], xyz[1, :], xyz[2, :]


def remap_shell(mhd, alt):
    '''
    Remap an MHD shell's values from its original altitude to a new altitude
    given in kilometers.
    Only affects 'x', 'y', and 'z' variables.
    '''

    from numpy import sin, cos, tan, arcsin, arccos, sqrt, isreal
    # Convert altitude to geocentric RE:
    r_new = 1.+alt/6371.0

    south = mhd['z'] < 0

    # Get mhd radiuses:
    r_old = sqrt(mhd['x']**2+mhd['y']**2+mhd['z']**2)
    xy_old = sqrt(mhd['x']**2+mhd['y']**2)
    xy_old[xy_old == 0.0] = 0.00001

    # Get MHD latitude:
    lat_old = arcsin(mhd['z']/r_old)

    # Get new latitude (see gombosi eq. 1.77):
    step = cos(lat_old) * sqrt(r_new/r_old)
    lat_new = arccos(step)
    # Convert sign:
    lat_new[south] *= -1

    # Get new z-values:
    mhd['z'] = r_new*sin(lat_new)

    xy_new = mhd['z']/tan(lat_new+0.00001)
    xy_ratio = xy_new/xy_old
    xy_ratio[~isreal(xy_ratio)] = 1.0

    mhd['x'] *= xy_ratio
    mhd['y'] *= xy_ratio

    mhd['r'] = sqrt(mhd['x']**2+mhd['y']**2+mhd['z']**2)


def pchip_fit(x, y, z, flux, nLons=45, dLat=90/51, debug=False):
    from scipy.interpolate import PchipInterpolator
    '''
    Divide the hemisphere into equal latitude bins. For each bin, find all
    points that fall within the latitude range [latnow:latnow+dLat].
    Then use a periodic PCHIP fit to interpolate the values *flux* from
    its original positions at *x*, *y*. A new longitude range is created
    using equally-spaced `nLons` points.

    Arg `z` is the z-coordinate in SM coordinates corresponding to `x` and `y`.
    It is used to calculate latitude/colatitude.

    Returns the new SM x and y values along with the interpolated flux values.
    '''

    # Output values:
    xOut, yOut, zOut = np.array([]), np.array([]), np.array([])

    # Polar coordinates:
    rad = np.sqrt(x**2 + y**2 + z**2)
    if rad.std() > 0.0001:
        raise ValueError('Varying Altitude detected.' +
                         'use constant altitude only.')
    rad = rad.mean()
    xylat, lon = np.sqrt(x**2+y**2), 180./np.pi * np.arctan2(y, x)
    colat = 180./np.pi * np.arctan2(xylat, z)

    if debug:
        fig, ax = plt.subplots(1, 1)
        ax.plot(xylat, colat, '.')
        ax.set_xlabel('XYLat')
        ax.set_ylabel('Colatitude')
        ax.set_title('Checking Coord Conversion')

    lonNew = np.linspace(-180, 180, nLons)
    lonRad = lonNew*np.pi/180.

    # Latitude bins:
    latbins = np.arange(colat.min(), colat.max(), dLat)

    if debug:
        print('Lat min and max = {}, {}'.format(colat.min(), colat.max()))

    for latnow in latbins:
        # Find points in current lat bin:
        loc = (colat >= latnow) & (colat < latnow+dLat)
        xNow = lon[loc]
        fluxNow = flux[loc]

        if fluxNow.size == 0:
            continue  # Require some points.

        # Ghost cells to enforce continuity:
        npts = fluxNow.size*3

        # Sort for victory:
        order = np.argsort(xNow)
        xNow = xNow[order]
        fluxNow = fluxNow[order]

        # Triplicate data to enforce periodicity.
        xLarge = np.reshape([xNow-360., xNow, xNow+360.], npts)
        zLarge = np.reshape([fluxNow, fluxNow, fluxNow], npts)

        # Create spline function:
        fit = PchipInterpolator(xLarge, zLarge)
        zFit = fit(lonNew)  # This is the interpolated flux!

        # Debug: show spline fitting.
        if debug and not np.all(zLarge == 0.0):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xLarge, zLarge, 'r.', label='Raw Data Points')
            ax.plot(lonNew, fit(lonNew), 'b-', label='PCHIP fit')
            ax.set_xlabel(r'SM Longitude ($^{\circ}$ from local noon)')
            ax.set_ylabel(r'Outflow Flux ($m^{-2}s^{-1}$)')
            ax.set_title('PCHIP Fitting: ' +
                         f'Colat=[{latnow:.3E}, {latnow + dLat:.3E}]')
            # ax.set_title('lat bin={:.3f} obs. range=[{:.3E},{:.3E}]'.format(
            #     latnow, zLarge.min(), zLarge.max()))
            fig.tight_layout()

        # Convert radius, colat back into SM X/Y:
        xynow = rad * np.sin(np.pi/180 * (latnow + .5*dLat))
        xOut = np.append(xOut, xynow*np.cos(lonRad))
        yOut = np.append(yOut, xynow*np.sin(lonRad))
        zOut = np.append(zOut, zFit)
        if debug:
            raise Exception
    # If things go spectacularly bad, just crash.
    if xOut.size == 0:
        raise ValueError('PCHIP produced zero real points')

    return xOut, yOut, zOut


def fix_axes(ax, rmax=3, fresh=True, dolabelx=True, title='', r_range=None):
    '''
    Set axes to be standard style.
    '''

    from matplotlib.patches import Circle

    ax.set_title(title, size=20)
    ax.set_axis_off()

    if not fresh:
        return

    if r_range is None:
        r_range = rmax

    # Add dark boundary around outermost point:
    circ = Circle((0, 0), r_range, fill=False, ec='k', lw=1.5)
    ax.add_artist(circ)

    # Add latitude circles.
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        r_text = r*np.sin(np.pi/4.0)
        if r > r_range:
            continue
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0, ls='dashed')
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('To Sun $\rightarrow')
    # if dolabely: ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*r_range, r_range])
    ax.set_ylim([-1.0*r_range, r_range])
    ax.hlines(0, -1.*r_range, r_range, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*r_range, r_range, colors='k', linestyles='dotted')


def read_shell_to_mhd(filename, irad=0):
    '''
    Open a ShellSlice object and perform converions to translate it into a
    shell object as used by ExHail analysis tools.

    irad sets the index of the radius to use; default is zero.
    '''

    from numpy import cos, sin
    from spacepy.pybats.bats import ShellSlice

    # Open file and calculate flux & fluence.
    mhd = ShellSlice(filename)
    mhd.calc_radflu('rho')  # both flux and fluence.

    # Calculate X, Y, and Z coordinates:
    rad = mhd['r'][irad]
    colat = mhd.theta
    z = rad * cos(colat)
    xy = rad * sin(colat)
    x = xy * cos(mhd.phi)
    y = xy * sin(mhd.phi)

    # Store flattened items into ShellSlice:
    mhd['x'] = x.flatten()
    mhd['y'] = y.flatten()
    mhd['z'] = z.flatten()
    mhd['flux'] = mhd['rho_rflx'].flatten()

    # Save fluence in same spot as expected:
    mhd['fluence'] = {'all': mhd['rho_rflu']}

    # Keep the regularly gridded fluence.
    mhd['flux_reg'] = mhd['rho_rflx'][irad, :, :].transpose()

    return mhd


class Satellite(PbData):
    '''
    A class that creates a single satellite on a circular orbit with its
    orbit plane rotated *theta* degrees counter-clockwise from local noon and
    tilted *phi* degrees from the magnetic pole.   A set of MHD results
    should be included as the *mhd* keyword.  See python module gmoutflow
    for details on opening and generating these items.
    '''

    def __init__(self, theta, phi, mhd, debug=False, npoints=401,
                 *args, **kwargs):
        from matplotlib import tri

        super(Satellite, self).__init__(*args, **kwargs)

        # Intialize the MHD data to work nicely with our needs here:
        # Start by creating triangulation and interpolator:
        if 'trip' not in mhd:
            posZ = mhd['z'] > 0  # Northern hemisphere only.
            triang = tri.Triangulation(mhd['x'][posZ], mhd['y'][posZ])
            mhd['trip'] = tri.LinearTriInterpolator(triang, mhd['flux'][posZ])

        # Get orbit radius from mhd file:
        self.rad = mhd['r'].max()

        # Create satellite trajectory, limit to northern hemisphere.
        x, y, z = circle(self.rad, phi, theta, npoints=npoints)
        loc = z > 0
        self['x'], self['y'], self['z'] = x[loc], y[loc], z[loc]

        # Use mhd file to get flux along trajectory:
        self['flux'] = np.array(mhd['trip'](x[loc], y[loc]))
        self['flux'][~np.isfinite(self['flux'])] = 0.0

    def add_flux_line(self, target=None, loc=111, cmap='RdBu_r', contour=None,
                      zlim=[-1E12, 1E12], add_cbar=False):
        '''
        Add a line colored by the amplitude of the radial flux to plot or
        figure designated by kwarg *target*

        If a contour object is given, the color map and plot limits are taken
        from that object.  Otherwise, the the kwargs *zlim* and *cmap*
        can be used to set these options.
        '''

        from matplotlib.colors import Normalize
        from matplotlib.collections import LineCollection

        fig, ax = set_target(target, loc=loc, figsize=(6, 6))

        # Try to use existing contour to get plotting information:
        if contour:
            cmap = contour.cmap
            norm = contour.norm
        else:
            if not zlim:
                raise ValueError('Either zlim or contour must be set')
            norm = Normalize(vmin=zlim[0], vmax=zlim[1])

        # Create line segments:
        points = np.array([self['x'], self['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a line collection and plot it.
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(self['flux'])
        lc.set_linewidth(3)
        ax.add_collection(lc)

        # Tidy up axes if brand new plot.
        if not issubclass(type(target), plt.Axes):
            fix_axes(ax, rmax=self.rad)

        return fig, ax, lc

    def add_orbit_line(self, target=None, loc=111, *args, **kwargs):
        '''
        Add the path of the orbit through the northern hemisphere to
        plot/figure *target*.  Extra args and kwargs are passed to
        matplotlib's plot command.
        '''

        fig, ax = set_target(target, loc=loc, figsize=(6, 6))
        line = ax.plot(self['x'], self['y'], *args, **kwargs)

        # Tidy up axes if brand new plot.
        if not issubclass(type(target), plt.Axes):
            fix_axes(ax, rmax=self.r)

        return fig, ax, line


class Constellation(PbData):
    '''
    A class that creates a constellation of *n_sat* satellites with circular
    orbit planes rotated *theta* degrees counter-clockwise from local noon and
    tilted *phi* degrees from the magnetic pole.  Both *theta* and *phi*
    arguments should be numpy arrays of size *n_sat*.  A set of MHD results
    should be included as the *mhd* keyword.  See python module gmoutflow
    for details on opening and generating these items.
    '''

    def __init__(self, theta, phi, mhd, debug=False, npoints=401,
                 *args, **kwargs):
        super(Constellation, self).__init__(*args, **kwargs)

        # Save that mhd data.
        self.mhd = mhd

        # Set debug flag:
        self.debug = debug

        # Get radius of plots/orbit from mhd:
        self['r'] = mhd['r'].max()
        if debug:
            fprint('constellation radius is {}'.format(self['r']))

        # Count our satellites:
        self.nsat = min(len(theta), len(phi))
        if debug:
            fprint('nsats = {}'.format(self.nsat))

        # Create the fake satellite trajectories:
        self['sats'] = []
        for t, p in zip(theta, phi):
            if debug:
                fprint('Adding satellite at {}, {}'.format(t, p))
            self['sats'].append(Satellite(t, p, self.mhd, npoints=npoints))

        # Calculate key values:
        if debug:
            fprint('Calculating flux...')
        self.calc_allflux()
        if debug:
            fprint('Calculating fluence...')
        self.calc_fluence()

    def calc_cc2d(self, colat_lim=40.):
        '''
        Calculate simple 2D correlation coefficient with no shifting.
        colat_lim is the latitude cutoff for the comparison and defaults to
        40 degrees (i.e., only points at or above 60 degrees magnetic latitude
        are considered).
        '''

        # print('WARNING: CC2D RESTRICTED TO HIGHER LATITUDES!  CHECK THIS!')
        # Create mask to only look at high latitude region:
        cutoff = colat_lim*np.pi/180.
        mask = self.theta <= cutoff

        return cc2d(self.mhd['flux_reg'][mask, :], self.flux_reg[mask, :])

    def calc_rms(self):
        '''
        Calculate and return the RMS error between the "real" (MHD result) and
        reconstructed 2D flux maps.
        '''

        err = self.flux_reg - self.mhd['flux_reg']

        return np.sqrt(np.sum(err**2)/err.size)

    def calc_corr2d_norm(self):
        '''

        '''
        from scipy.signal import correlate2d

        return correlate2d(self.flux_reg, self.mhd['flux_reg'])[0, 0]

    def calc_allflux(self):
        '''
        Using PCHIP fits over many latitudes, create 2D flux.
        '''

        xAll, yAll, zAll = np.array([]), np.array([]), np.array([])
        fAll = np.array([])  # All fluxes.
        for s in self['sats']:
            xAll = np.append(xAll, s['x'])
            yAll = np.append(yAll, s['y'])
            zAll = np.append(zAll, s['z'])
            fAll = np.append(fAll, s['flux'])

        self.xAll = xAll
        self.yAll = yAll
        self.zAll = zAll
        self.fAll = fAll

        self['x'], self['y'], self['flux'] = \
            pchip_fit(xAll, yAll, zAll, fAll, debug=self.debug)

        # Save number of lats and lons:
        self.attrs['nlon'] = 45  # same as value from pchip_fit, default=45
        self.attrs['nlat'] = self['flux'].size / self.attrs['nlon']

    def calc_fluence(self):
        '''
        Get fluence for both the MHD (saved as self['flu_mhd']) and for the
        "observations" made by the virtual satellite constellation (saved as
        self['flu']).
        '''
        from matplotlib.tri import Triangulation, LinearTriInterpolator
        from gmoutflow import calc_fluence

        # get fluence from MHD file.  Go ahead and stash it in the
        # original object to avoid recalculations.
        if 'fluence' not in self.mhd:
            self.mhd['fluence'] = calc_fluence(self.mhd)[0]
        self['flu_mhd'] = self.mhd['fluence']['all']

        # Now, calculate fluence for interpolated observed values:
        # Start with some good constants:
        R = self['r'] * 6371.0  # radius in km.
        # Match MHD values point-to-point by using same spacing:
        if hasattr(self.mhd, 'dtheta'):
            # ShellSlice objects are easy.
            dTheta = self.mhd.dtheta
            dPhi = self.mhd.dphi
            theta = self.mhd.theta[0, 0, :]
            phi = self.mhd.phi[0, :, 0]
        else:
            # Otherwise, assume old spacing from postproc'd shells:
            dTheta = 1.0 * np.pi/180.  # 1.0-degree spacing
            dPhi = 2.5 * np.pi/180.  # 2.5-degree spacing
            theta = np.arange(0, np.pi/2.0, dTheta)
            phi = np.arange(0, 2.*np.pi,  dPhi)

        # Create evenly-spaced grid.
        x, y = np.zeros(len(theta)*len(phi)), np.zeros(len(theta)*len(phi))
        theta_all = np.zeros(len(theta)*len(phi))
        phi_all = np.zeros(len(theta)*len(phi))
        nphi = len(phi)
        for i in range(len(theta)):
            theta_all[i*nphi:(i+1)*nphi] = theta[i]
            phi_all[i*nphi:(i+1)*nphi] = phi
            x[i*nphi:(i+1)*nphi] = R*np.sin(theta[i])*np.cos(phi)
            y[i*nphi:(i+1)*nphi] = R*np.sin(theta[i])*np.sin(phi)

        # Interpolate from original, irregular grid onto our nice evenly spaced
        # grid.  Do this using 2D Delaunay natural neighbor, which, for our
        # large number of points, works just fine.
        tri = Triangulation(self['x']*6371.0, self['y']*6371.0)
        interp = LinearTriInterpolator(tri, self['flux'])
        flux = interp(x, y)

        # floor out bad values.
        flux[~np.isfinite(flux)] = 0.0
        Flu = R**2 * flux * np.sin(theta_all) * dTheta * dPhi * 1E6

        # Limit result to colats where there is satellite coverage:
        xy = np.sqrt(self['x']**2+self['y']**2)
        min_colat = np.arcsin(xy.min()/self['r'])

        self['flu'] = Flu[theta_all > min_colat].sum()

        if self.debug:
            print('DEBUG:\n\tMin. Colat={}\n\tFluence={}\n\tLimited={}'.format(
                min_colat*180/np.pi, Flu.sum(), self['flu']))

        # Save the evenly-spaced data within object:
        self.theta, self.phi = theta, phi
        self.flux_reg = np.reshape(flux, (theta.size, phi.size))

    def add_flux_obs(self, target=None, loc=111, cmap='RdBu_r', contour=None,
                     zmax=1E12, add_cbar=False, nlevs=101, colat_max=None):
        '''
        Add a total flux plot derived from "observations" made by the
        virtual satellite constellation.

        If a contour object is given, the color map and plot limits are taken
        from that object.  Otherwise, the the kwargs *zlim* and *cmap*
        can be used to set these options.
        '''

        from matplotlib.colors import Normalize
        from matplotlib.patches import Ellipse

        fig, ax = set_target(target, loc=loc, figsize=(6, 6))

        # Try to use existing contour to get plotting information:
        if contour:
            cmap = contour.cmap
            norm = contour.norm
            levels = contour.levels
        else:
            if not zmax:
                raise ValueError('Either zlim or contour must be set')
            levels = np.linspace(-1.*zmax, zmax, nlevs)
            norm = Normalize(vmin=-1.*zmax, vmax=zmax)

        # Determine maximum colat in terms of plot radius:
        rAll = np.sqrt(self['x']**2+self['y']**2)
        if colat_max is None:
            r_range = rAll.max()  # include all points.
        else:
            r_range = rAll.max()*np.sin(np.pi/180.*colat_max)

        # Block out high-colat points, create contour.
        loc = rAll <= r_range
        cnt = ax.tricontourf(self['x'][loc], self['y'][loc], self['flux'][loc],
                             levels, cmap=cmap, norm=norm, extend='both')

        # Determine how well we are covering the polar cap.
        # Use a white circle to blank out polar cap if not covered.
        if rAll.min() > (np.pi/36):  # min colat>5deg, add circle:
            radius = 2*.95*rAll.min()
            circ = Ellipse((0, 0), radius, radius, fc='w', ec='w', alpha=.9)
            ax.add_artist(circ)

        # Tidy up axes if brand new plot.
        if not issubclass(type(target), plt.Axes):
            fix_axes(ax, rmax=self['r'], r_range=r_range)
        return fig, ax, cnt

    def add_flux_lines(self, target=None, loc=111, add_cbar=False,
                       colat_max=None, **kwargs):
        '''
        For each satellite in the collection, add a colored line representing
        the "observed" flux
        '''

        fig, ax = set_target(target, loc=loc, figsize=(6, 6))

        # Determine maximum colat in terms of plot radius:
        rAll = np.sqrt(self['x']**2+self['y']**2)
        if colat_max is None:
            r_range = rAll.max()  # include all points.
        else:
            r_range = rAll.max()*np.sin(np.pi/180.*colat_max)

        for s in self['sats']:
            s.add_flux_line(target=ax, **kwargs)

        # Tidy up axes if brand new plot.
        if not issubclass(type(target), plt.Axes):
            fix_axes(ax, rmax=self['r'], r_range=r_range)

        return fig, ax

    def add_orbit_lines(self, target=None, loc=111, add_cbar=False,
                        *args, **kwargs):
        '''
        For each satellite in the collection, add the path of the orbit through
        the northern hemisphere to the plot target.
        '''
        fig, ax = set_target(target, loc=loc, figsize=(6, 6))

        for s in self['sats']:
            s.add_orbit_line(target=ax, *args, **kwargs)

        # Tidy up axes if brand new plot.
        if not issubclass(type(target), plt.Axes):
            fix_axes(ax, rmax=self['r'])

        return fig, ax

    def plot_orbits_3d(self):
        '''
        Create a quick-look 3D plot of the orbit paths.
        '''
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for s in self['sats']:
            ax.plot(s['x'], s['y'], s['x'])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


if __name__ == '__main__':
    '''
    Run some demonstrations.
    '''

    import gmoutflow as gmo

    # Demonstration 1: Circular satellite orbits.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for a in [0, 45, 90, 135]:
        x, y, z = circle(1, 10, a)
        ax.plot(x, y, z, label='Azimuth={}'.format(a)+r'$^{\circ}$')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    # Demonstration 2: Example of a flux reconstruction
    # Open an MHD shell file (legacy format):
    mhd = gmo.read_shell('sample_data/shell_r300_southward.dat')
    remap_shell(mhd, 1400)

    # Fly a satellite constellation through it:
    nsats = 5
    azim_all = [x for x in np.linspace(-45, 45, nsats)]
    sats = Constellation(azim_all, nsats*[10], mhd)

    # Create figure:
    fig = plt.figure(figsize=[9, 4])

    # Add flux plot, show where sats are flying:
    fig, a1, cont = gmo.add_flux_plot(mhd, target=fig, loc=131,
                                      add_cbar=False, colat_max=45)
    sats.add_orbit_lines(target=a1)
    # Show flux extractions:
    f, a2 = sats.add_flux_lines(target=fig, loc=132, colat_max=45)
    # Show reconstruction:
    f, a3, cont = sats.add_flux_obs(target=fig, loc=133, colat_max=45)

    # Clean up and label:
    a2.set_title('Virtual Obs.')
    a3.set_title('Reconstruction')
    fig.tight_layout()

    plt.show()
