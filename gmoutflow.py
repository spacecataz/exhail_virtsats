#!/usr/bin/env python3
'''
Tools for doing the GM Outflow study.
'''

import numpy as np

# Here is our MHD varmap for converting Tecplot var names
# to more user-friendly versions:
varmap = {r"X [R]": 'x', r"Y [R]": 'y', r"Z [R]": 'z', r"`r [amu/cm^3]": 'rho',
          r"U_x [km/s]": 'ux', r"U_y [km/s]": 'uy',  r"U_z [km/s]": 'uz',
          r"B_x [nT]": 'bx', r"B_y [nT]": 'by', r"B_z [nT]": 'bz',
          r"Pe [nPa]": 'pe', r"p [nPa]": 'p',
          r"`r^S^w [amu/cm^3]": 'rhoSw', r"p^S^w [nPa]": 'pSw',
          r"U_x^S^w [km/s]": 'uxSw', r"U_y^S^w [km/s]": 'uySw',
          r"U_z^S^w [km/s]": 'uzSw',
          r"`r^H^p [amu/cm^3]": 'rhoHp', r"p^H^p [nPa]": 'pHp',
          r"U_x^H^p [km/s]": 'uxHp', r"U_y^H^p [km/s]": 'uyHp',
          r"U_z^H^p [km/s]": 'uzHp',
          r"`r^O^p [amu/cm^3]": 'rhoOp', r"p^O^p [nPa]": 'pOp',
          r"U_x^O^p [km/s]": 'uxOp', r"U_y^O^p [km/s]": 'uyOp',
          r"U_z^O^p [km/s]": 'uzOp',
          r"J_x [`mA/m^2]": 'jx', r"J_y [`mA/m^2]": 'jy',
          r"J_z [`mA/m^2]": 'jz', r"rad": 'r',
          r"X Pressure Gradient": 'gradpx',
          r"Y Pressure Gradient": 'gradpy',
          r"Z Pressure Gradient": 'gradpz'}


def create_stream(mhd, theta, phi, rstart=2.7, rstop=5.0):
    '''
    Given an Y=0 MHD slice, *mhd*, and starting lat/lon *theta* and *phi*,
    create a stream trace out to R=5RE, extract all variables, and create
    a data object analogous to that created by *read_tecstream_short*.

    Both theta and phi are given in degrees. Phi should only be 0 or 180
    because traces are restricted to the Y=0 plane.
    '''

    # Ensure all variables are in our data object.
    if 'gradpr' not in mhd:
        mhd['u'] = np.sqrt(mhd['ux']**2 + mhd['uy']**2 + mhd['uz']**2)
        mhd.calc_gradP()
        mhd.calc_divmomen()
        mhd['r'] = np.sqrt(mhd['x']**2 + mhd['z']**2)
        ratio = mhd['z']/mhd['r']
        ratio[ratio > 1.] = 1.0
        ratio[ratio < -1.] = -1.0
        mhd['theta'] = np.arccos(ratio)
        mhd['phi'] = np.arctan2(0, mhd['x'])
        mhd['ur'] = mhd['ux']*np.sin(mhd['theta'])*np.cos(mhd['phi']) + \
            mhd['uy']*np.sin(mhd['theta'])*np.sin(mhd['phi']) + \
            mhd['uz']*np.cos(mhd['theta'])
        mhd['gradpr'] = \
            mhd['gradP_x']*np.sin(mhd['theta'])*np.cos(mhd['phi']) + \
            mhd['gradP_z']*np.cos(mhd['theta'])
        # JxB force; radial JxB force.
        mhd.calc_jxb()
        mhd['jbr'] = \
            mhd['jbx']*np.sin(mhd['theta'])*np.cos(mhd['phi']) + \
            mhd['jby']*np.sin(mhd['theta'])*np.sin(mhd['phi']) + \
            mhd['jbz']*np.cos(mhd['theta'])
        # Momentum divergence.
        mhd['divr'] = \
            mhd['divmomx']*np.sin(mhd['theta'])*np.cos(mhd['phi']) + \
            mhd['divmomz']*np.cos(mhd['theta'])
        # mhd['divr']*=-1.

        # Total force density.
        mhd['f_total'] = mhd['gradpr']+mhd['jbr']+mhd['divr']

    xstart = rstart * np.sin(theta*np.pi/180.) * np.cos(phi*np.pi/180.)
    zstart = rstart * np.cos(theta*np.pi/180.)
    trace = mhd.get_stream(xstart, zstart, 'ux', 'uz')

    # Reduce stream to R=[rstart, rstop].
    index = np.arange(trace.x.size)  # array of indices.
    r = np.sqrt(trace.x**2 + trace.y**2)

    i1, i2 = index[r >= rstart][0], index[r >= rstop][0]
    x, y = trace.x[i1:i2], trace.y[i1:i2]

    # Now, extract values along trace.
    data = mhd.extract(x, y)

    # Do integration.
    npts = data['x'].size
    data['dS'] = np.zeros(npts)
    data['dS'][1:] = np.sqrt(np.ediff1d(data['x'])**2 +
                             np.ediff1d(data['z'])**2) * 6371.0  # RE->km
    data['dt'] = np.zeros(npts)
    data['dt'] = data['dS']/data['u']  # T in seconds.
    data['t'] = np.cumsum(data['dt'])

    # Create some conversion constants.
    force = 1E-9  # nN      -> N
    mass = 1.6726E-21  # AMU/cm3 -> kg/m3
    grav = -6.672E-11*5.972E24  # m3/s2
    factor = data['dt']*force/(data['rho']*mass)
    # A = 1E-12/1.6726E-21
    data['gravr'] = data['rho']*mass*grav / (data['r']*6371000.0)**2/force

    # Integrate.
    data['u_p'] = np.cumsum(data['gradpr'] * factor)/1000.0
    data['u_j'] = np.cumsum(data['jbr'] * factor)/1000.0
    data['u_g'] = np.cumsum(grav/(data['r']*6371000.0)**2*data['dt'])/1000.0
    data['u_d'] = np.cumsum(data['divr'] * factor)/1000.0

    data['ux_p'] = np.cumsum(data['gradP_x'] * factor)/1000.0
    data['ux_j'] = np.cumsum(data['jbx'] * factor)/1000.0
    data['ux_d'] = np.cumsum(data['divmomx'] * factor)/1000.0

    data['uz_p'] = np.cumsum(data['gradP_z'] * factor)/1000.0
    data['uz_j'] = np.cumsum(data['jbz'] * factor)/1000.0
    data['uz_d'] = np.cumsum(data['divmomz'] * factor)/1000.0

    # Going the other way: F_r from U_r.
    dUr = np.zeros(npts)
    dUr[1:] = np.ediff1d(data['ur']*1000.0)
    data['F_r'] = dUr*mass*data['rho']/data['dt']/force
    data['F_r'][0] = 0

    # Create useful sums.
    data['F_sum'] = data['gradpr']+data['jbr']+data['gravr']+data['divr']
    data['U_sum'] = data['u_j']+data['u_p']+data['u_g']+data['u_d']

    # Store some meta-data.
    data.attrs['theta'], data.attrs['phi'] = theta, phi

    return data


def read_tecstream_short(filename):

    # Open file, skip header.
    f = open(filename, 'r')
    line = ''
    while line[0:5] != ' DT=(':
        line = f.readline()

    # Slurp remainder of file.
    lines = f.readlines()
    f.close()

    # Create data container.
    var = ['x', 'y', 'z', 'rho', 'ux', 'uy', 'uz', 'bx', 'by', 'bz', 'p',
           'jx', 'jy', 'jz', 'r', 'gradpx', 'gradpy', 'gradpz']
    nLines = len(lines)
    data = {}
    for v in var:
        data[v] = np.zeros(nLines)
    # data['r'] = np.zeros(nLines)

    for i, l in enumerate(lines):
        parts = l.split()
        for v, p in zip(var, parts):
            data[v][i] = p
        # data['r'][i] = data['x'][i]**2 + data['y'][i]**2 + data['z'][i]**2

    # Trim out points that lie well inside the inner boundary.
    loc = data['r'] >= 2.65
    for v in var:
        data[v] = data[v][loc]

    # ## DATA CALCULATIONS ###
    # Simple calculations:
    # data['r'] = np.sqrt(data['r'])
    data['u'] = np.sqrt(data['ux']**2+data['uy']**2+data['uz']**2)
    data['temp'] = data['p']/data['rho'] * 6.24150935
    data['filename'] = filename

    # Spherical Coords:
    ratio = data['z']/data['r']
    ratio[ratio > 1.] = 1.0
    ratio[ratio < -1.] = -1.0
    data['theta'] = np.arccos(ratio)
    data['phi'] = np.arctan2(data['y'], data['x'])
    # Radial velocity:
    data['ur'] = \
        data['ux']*np.sin(data['theta'])*np.cos(data['phi']) + \
        data['uy']*np.sin(data['theta'])*np.sin(data['phi']) + \
        data['uz']*np.cos(data['theta'])
    # Radial pressure gradient:
    conv = -1./6371000.0  # nN/m^2/Re -> nN/m^3
    data['gradpr'] = \
        data['gradpx']*np.sin(data['theta'])*np.cos(data['phi']) + \
        data['gradpy']*np.sin(data['theta'])*np.sin(data['phi']) + \
        data['gradpz']*np.cos(data['theta'])
    data['gradpr'] *= conv
    # JxB force; radial JxB force.
    factor = 1E-6  # milliAmps->Amps
    data['jbx'] = factor*(data['jy']*data['bz'] - data['jz']*data['by'])
    data['jby'] = factor*(data['jz']*data['bx'] - data['jx']*data['bz'])
    data['jbz'] = factor*(data['jx']*data['by'] - data['jy']*data['bx'])
    data['jbr'] = \
        data['jbx']*np.sin(data['theta'])*np.cos(data['phi']) + \
        data['jby']*np.sin(data['theta'])*np.sin(data['phi']) + \
        data['jbz']*np.cos(data['theta'])
    # Total force density.
    data['f_total'] = data['gradpr']+data['jbr']

    # Finally, integrate forces along streamline.
    npts = data['x'].size
    data['dS'] = np.zeros(npts)
    data['dS'][1:] = np.sqrt(np.ediff1d(data['x'])**2 +
                             np.ediff1d(data['y'])**2 +
                             np.ediff1d(data['z'])**2) * 6371.0  # RE->km
    data['dt'] = np.zeros(npts)
    data['dt'] = data['dS']/data['u']  # T in seconds.
    data['t'] = np.cumsum(data['dt'])

    # Create some conversion constants.
    force = 1E-9  # nN      -> N
    mass = 1.6726E-21  # AMU/cm3 -> kg/m3
    grav = -6.672E-11*5.972E24  # m3/s2
    factor = data['dt']*force/(data['rho']*mass)
    # A = 1E-12/1.6726E-21
    data['gravr'] = data['rho']*mass*grav / (data['r']*6371000.0)**2/force

    # Integrate.
    data['u_p'] = np.cumsum(data['gradpr'] * factor)/1000.0
    data['u_j'] = np.cumsum(data['jbr'] * factor)/1000.0
    data['u_g'] = np.cumsum(grav / (data['r']*6371000.0)**2*data['dt'])/1000.0

    # Going the other way: F_r from U_r.
    dUr = np.zeros(npts)
    dUr[1:] = np.ediff1d(data['ur']*1000.0)
    data['F_r'] = dUr*mass*data['rho']/data['dt']/force
    data['F_r'][0] = 0

    # Create useful sums.
    data['F_sum'] = data['gradpr']+data['jbr']+data['gravr']
    data['U_sum'] = data['u_j']+data['u_p']+data['u_g']

    return data


def force_check(data):
    '''
    Do a sanity-check calculation using mean values.
    '''

    t = data['t'][-1]
    Fg = data['gravr'].mean()
    Fp = data['gradpr'].mean()
    Fj = data['jbr'].mean()
    Fd = data['divr'].mean()
    rho = data['rho'].mean()
    A = t * 1E-12 / (rho*1.6726E-21)  # Conversion factor; yields km/s.

    fmt = '{:10s}\t{:10.3E}\t{:+10.2f}'
    print('Time={:.2f}s, Density={:.2f}#/cm3.'.format(t, rho))
    print('{:10s}\t{:10s}\t{:10s}'.format(
            'Term', 'Force (nN/m3)', 'Velocity (km/s)'))
    print(fmt.format('Gravity', Fg, A*Fg))
    print(fmt.format('GradP',   Fp, A*Fp))
    print(fmt.format('JxB',     Fj, A*Fj))
    print(fmt.format('div-p',   Fd, A*Fd))
    print(fmt.format('Total',   Fg+Fp+Fj, A*(Fg+Fp+Fj)))
    print(fmt.format('MHD', data['F_r'].mean(), A*data['F_r'].mean()))


def force_quicklook(data):
    '''
    From a short-stream data object (with all forces and velocities calculated)
    plot the time evolution of the fluid parcel.
    '''

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))

    def fix_ax(ax, ylabel, ncol=1):
        ax.grid()
        ax.set_xlabel('Time ($s$)')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', ncol=ncol)

    # Position vs. Time
    a1 = fig.add_subplot(311)
    a1.plot(data['t'], data['x'], 'k--', label='GSM X')
    a1.plot(data['t'], data['z'], 'k-.', label='GSM Z')
    a1.plot(data['t'], data['r'], 'k-',  label='Geocentric $R$')
    if hasattr(data, 'attrs'):
        a1.set_title(
            r'Stream started from $\Theta={:4.1f}$, $\phi={:4.0f}$'.format(
                data.attrs['theta'],
                data.attrs['phi'])+'\n'+data.attrs['file'])
        # fig.suptitle(data.attrs['file'])
    fix_ax(a1, '$R_E$')
    # Force breakdown vs. Time
    a2 = fig.add_subplot(312)
    a2.plot(data['t'], data['F_r'],    'k-.', label=r'$F_R$')
    a2.plot(data['t'], data['gradpr'], 'r', label=r'$\nabla P$')
    a2.plot(data['t'], data['jbr'],    'b', label=r'$J\times B$')
    a2.plot(data['t'], data['divr'],   'g', label=r'$\rho(u\cdot\nabla)u$')
    # a2.plot(data['t'], data['gravr'],  'g', label=r'$G$')
    a2.plot(data['t'], data['F_sum'],
            'k', label=r'$\Sigma F$')
    fix_ax(a2, '$nN/m^3$', ncol=2)
    # Velocity Breakdown vs. Time
    a3 = fig.add_subplot(313)
    # a3.plot(data['t'], data['u'],   'k--', label=r'$|U|$')
    a3.plot(data['t'], data['ur'],  'k-.', label=r'$U_R$')
    a3.plot(data['t'], data['u_p'], 'r',   label=r'$U_{\nabla P}$')
    a3.plot(data['t'], data['u_j'], 'b',   label=r'$U_{J\times B}$')
    a3.plot(data['t'], data['u_d'], 'g',   label=r'$\rho(u\cdot\nabla)u$')
    a3.plot(data['t'], data['U_sum'], 'k-',
            label=r'$U_{\Sigma U}$')
    fix_ax(a3, '$km/s$', ncol=2)

    fig.tight_layout()


def parse_vars(filename):
    '''
    Map variables from awkward Tecplot names to something easier to handle.
    '''
    import re
    var = []

    with open(filename, 'r') as f:
        line = ''
        # Jump to start of variables:
        while 'VARIABLES' not in line:
            line = f.readline()
        v = re.search('\"(.*)\"', line).groups()[0]
        var.append(varmap[v])

        for line in f:
            if line[0] != '"':
                break
            match = re.search('\"(.*)\"', line)
            try:
                var.append(varmap[match.groups()[0]])
            except KeyError:
                print(f'Unknown variable: {line}')
    return var


def read_shell(infile, debug=False):
    '''
    Read a tecplot ascii 2D shell extraction file.
    Return a dictionary of variable name-array pairs.
    '''

    # var = ['x', 'y', 'z', 'rho', 'ux', 'uy', 'uz', 'bx', 'by', 'bz', 'p',
    #       'jx', 'jy','jz', 'r', 'gradpx', 'gradpy', 'gradpz']

    # Parse file header:
    var = parse_vars(infile)

    # Open file, skip header.
    f = open(infile, 'r')
    line = ''
    while line[0:5] != ' DT=(':
        line = f.readline()

    # Slurp remainder of file.
    lines = f.readlines()
    f.close()
    maxlines = len(lines)
    if debug:
        print("Found {:d} lines in the file.".format(maxlines))

    # Create containers.
    data = {}
    for v in var:
        data[v] = np.zeros(maxlines)

    # Parse data until we hit that crazy number stack at the end.
    iLine = 0
    for l in lines:
        parts = l.split()
        if len(parts) != len(var):
            if debug:
                print("Stopping at line #%i" % (iLine+1))
                print("...which reads, ", l)
            break
        for (p, v) in zip(parts, var):
            data[v][iLine] = p
        # Iterate line numbers.
        iLine += 1

    if debug:
        print("Finished reading, found %i good lines of %i possible" %
              (iLine, maxlines))
    # Trim back arrays to get rid of unused space.
    for v in var:
        data[v] = data[v][:iLine]

    # ##Calculate additional variables.
    ratio = data['z']/data['r']
    ratio[ratio > 1.] = 1.0
    ratio[ratio < -1.] = -1.0
    # Coords:
    data['theta'] = np.arccos(ratio)
    data['phi'] = np.arctan2(data['y'], data['x'])
    # Radial velocity:
    data['ur'] = \
        data['ux']*np.sin(data['theta'])*np.cos(data['phi']) + \
        data['uy']*np.sin(data['theta'])*np.sin(data['phi']) + \
        data['uz']*np.cos(data['theta'])
    for s in ['Sw', 'Hp', 'Op']:
        if 'ux'+s not in var:
            continue
        data['ur'+s] = \
            data['ux'+s]*np.sin(data['theta'])*np.cos(data['phi']) + \
            data['uy'+s]*np.sin(data['theta'])*np.sin(data['phi']) + \
            data['uz'+s]*np.cos(data['theta'])

    # Radial pressure gradient:
    conv = -1./6371000.0  # nPa/m^2/Re -> nPa/m^3
    data['gradpr'] = \
        data['gradpx']*np.sin(data['theta'])*np.cos(data['phi']) + \
        data['gradpy']*np.sin(data['theta'])*np.sin(data['phi']) + \
        data['gradpz']*np.cos(data['theta'])
    data['gradpr'] *= conv
    # Radial flux:
    factor = 1000. * (100.0)**3  # km->m, cm^-3->m^-3
    for s in ['', 'Sw', 'Hp', 'Op']:
        if 'ux'+s not in var:
            continue
        data['flux'+s] = factor * data['ur'+s] * data['rho'+s] / \
            (1+15*(s == 'Op'))
    # JxB force; radial JxB force.
    factor = 1E-6  # milliAmps->Amps
    data['jbx'] = factor*(data['jy']*data['bz'] - data['jz']*data['by'])
    data['jby'] = factor*(data['jz']*data['bx'] - data['jx']*data['bz'])
    data['jbz'] = factor*(data['jx']*data['by'] - data['jy']*data['bx'])
    data['jbr'] = \
        data['jbx']*np.sin(data['theta'])*np.cos(data['phi']) + \
        data['jby']*np.sin(data['theta'])*np.sin(data['phi']) + \
        data['jbz']*np.cos(data['theta'])
    # Total force density.
    data['f_total'] = data['gradpr']+data['jbr']

    # Return data to caller.
    return data


def shell_recalc(data):
    '''
    Given a shell dictionary as created by the function *read_shell*,
    re-calculate variables such as fluxes, current, etc.
    '''

    return True


def add_flux_plot(data, s='', target=None, zmax=2e12, zmin=1e8, loc=111,
                  dolabel=True, add_cbar=False, dolabelx=True, dolabely=True,
                  dofill=True, nlevs=50, colat_max=None, showax=False,
                  ntick=4):
    '''
    Create a dial plot of radial outflow fluxes and place it onto
    *target*, where *target* is either a figure, axes, or **None** (in which
    case a new figure will be generated.)
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.ticker import (MultipleLocator, ScalarFormatter)

    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc)
    elif issubclass(type(target), plt.Axes):
        ax = target
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(loc)

    # Create levels.
    fmt = ScalarFormatter(useOffset=False, useMathText=True)
    ticks = MultipleLocator(zmax/ntick)
    levs = np.linspace(-1.*zmax, zmax, nlevs)

    # Determine maximum colat in terms of plot radius:
    rmax = data['r'][0]
    if colat_max is None:
        r_range = rmax  # include all points.
    else:
        r_range = rmax*np.sin(np.pi/180.*colat_max)

    # Add contour.
    loc = (data['z'] > 0)
    flux = data['flux'+s][loc]

    if dofill:
        func = ax.tricontourf
    else:
        func = ax.tricontour

    cnt = func(data['x'][loc], data['y'][loc],
               flux, levs, cmap=plt.get_cmap('RdBu_r'), extend='both')

    # Add colorbar.
    if add_cbar:
        cb = plt.colorbar(cnt, ax=ax, shrink=.85, extend='both', ticks=ticks,
                          format=fmt)
        cb.set_label(r'$Flux$ ($m^{-2}s^{-1}$)')

    # Add latitude circles.
    # Start with boundary of code:
    circ = Circle((0, 0), r_range, fill=False, ec='k', lw=1.5, zorder=5)
    ax.add_artist(circ)
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        if r > r_range:
            continue
        r_text = r*np.sin(np.pi/4.0)
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0,
                      ls='dashed', zorder=5)
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('GSM X ($R_E$)')
    if dolabely:
        ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*r_range, r_range])
    ax.set_ylim([-1.0*r_range, r_range])
    ax.hlines(0, -1.*r_range, r_range, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*r_range, r_range, colors='k', linestyles='dotted')
    if not showax:
        ax.set_axis_off()
    if dolabel:
        ax.set_title('Shell at R=%4.2f$R_E$' % rmax)

    return fig, ax, cnt


def add_ur_plot(data, s='', target=None, zmax=50, loc=111, dolabel=True,
                add_cbar=False, dolabelx=True, dolabely=True, showax=True,
                ntick=10):
    '''
    Create a dial plot of radial outflow velocity and place it onto
    *target*, where *target* is either a figure, axes, or **None** (in which
    case a new figure will be generated.)

    Note that the first kwarg, *s*, sets the species.  It defaults to
    no value (i.e., the total plasma Ur), but can be set to a species name
    (e.g., 'Sw' or 'Op') to get a species-specific plot.
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.ticker import MultipleLocator as ML

    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc)
    elif type(target).__base__ == plt.Axes:
        ax = target
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(loc)

    # Create levels.
    levs = np.linspace(-1.*zmax, zmax)

    # Add contour.
    cnt = ax.tricontourf(data['x'][data['z'] > 0], data['y'][data['z'] > 0],
                         data['ur'+s][data['z'] > 0], levs, extend='both',
                         cmap=plt.get_cmap('RdBu_r'))
    # Add colorbar.
    if add_cbar:
        cb = plt.colorbar(cnt, ax=ax, shrink=.75, extend='both',
                          ticks=ML(ntick))
        cb.set_label(s+bool(s)*' '+r'$U_{Radial}$ ($km/s$)')
    else:
        cb = False

    # Add latitude circles.
    rmax = data['r'][0]
    circ = Circle((0, 0), rmax, fill=False, ec='k', lw=1.5)
    ax.add_artist(circ)
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        r_text = r*np.sin(np.pi/4.0)
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0, ls='dashed')
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('GSM X ($R_E$)')
    if dolabely:
        ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*rmax-0.05, rmax+0.05])
    ax.set_ylim([-1.0*rmax-0.05, rmax+0.05])
    ax.hlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    if dolabel:
        ax.set_title('Shell at R=%4.2f$R_E$' % rmax)
    if not showax:
        ax.set_axis_off()
    return fig, ax, cnt


def add_force_plot(data, target=None, zmax=1E-6, loc=111, dolabel=True,
                   add_cbar=False, dolabelx=True, dolabely=True):
    '''
    Create a dial plot of total radial forces and place it onto
    *target*, where *target* is either a figure, axes, or **None** (in which
    case a new figure will be generated.)
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.ticker import (MultipleLocator, ScalarFormatter)

    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc)
    elif type(target).__base__ == plt.Axes:
        ax = target
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(loc)

    # Create levels, norms.
    levs = np.linspace(-1*zmax, zmax, 51)

    # Add contour.
    gradpr = data['f_total'][data['z'] > 0]

    # Two contours: one for up, one for down.
    cn1 = ax.tricontourf(data['x'][data['z'] > 0], data['y'][data['z'] > 0],
                         gradpr, levs, cmap=plt.get_cmap('PuOr_r'),
                         extend='both')

    # Add colorbar.
    if add_cbar:
        fmt = ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_powerlimits((-4, 3))
        ticks = MultipleLocator(zmax/4)
        cb = plt.colorbar(cn1, ax=ax, shrink=.85, ticks=ticks, format=fmt)
        cb.set_label(r'$F_{radial}$ ($nN/m^{3}$)')

    # Add latitude circles.
    rmax = data['r'][0]
    circ = Circle((0, 0), rmax, fill=False, ec='k', lw=1.5)
    ax.add_artist(circ)
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        r_text = r*np.sin(np.pi/4.0)
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0, ls='dashed')
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('GSM X ($R_E$)')
    if dolabely:
        ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*rmax, rmax])
    ax.set_ylim([-1.0*rmax, rmax])
    ax.hlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    if dolabel:
        ax.set_title('Shell at R=%4.2f$R_E$' % rmax)

    return fig, ax, cn1


def add_jbr_plot(data, target=None, zmax=1E-6, loc=111, dolabel=True,
                 add_cbar=False, dolabelx=True, dolabely=True):
    '''
    Create a dial plot of radial JxB (force) and place it onto
    *target*, where *target* is either a figure, axes, or **None** (in which
    case a new figure will be generated.)
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.ticker import (MultipleLocator, ScalarFormatter)

    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc)
    elif type(target).__base__ == plt.Axes:
        ax = target
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(loc)

    # Create levels, norms.
    levs = np.linspace(-1*zmax, zmax, 51)

    # Add contour.
    gradpr = data['jbr'][data['z'] > 0]

    # Two contours: one for up, one for down.
    cn1 = ax.tricontourf(data['x'][data['z'] > 0], data['y'][data['z'] > 0],
                         gradpr, levs, cmap=plt.get_cmap('PuOr_r'),
                         extend='both')

    # Add colorbar.
    if add_cbar:
        fmt = ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_powerlimits((-4, 3))
        ticks = MultipleLocator(zmax/4)
        cb = plt.colorbar(cn1, ax=ax, shrink=.85, ticks=ticks, format=fmt)
        cb.set_label(r'$J \times B$ ($nN/m^{3}$)')

    # Add latitude circles.
    rmax = data['r'][0]
    circ = Circle((0, 0), rmax, fill=False, ec='k', lw=1.5)
    ax.add_artist(circ)
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        r_text = r*np.sin(np.pi/4.0)
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0, ls='dashed')
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('GSM X ($R_E$)')
    if dolabely:
        ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*rmax, rmax])
    ax.set_ylim([-1.0*rmax, rmax])
    ax.hlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    if dolabel:
        ax.set_title('Shell at R=%4.2f$R_E$' % rmax)

    return fig, ax, cn1


def add_gradpr_plot(data, target=None, zmax=1E-6, loc=111, dolabel=True,
                    add_cbar=False, dolabelx=True, dolabely=True):
    '''
    Create a dial plot of radial pressure gradient (force) and place it onto
    *target*, where *target* is either a figure, axes, or **None** (in which
    case a new figure will be generated.)
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.ticker import (MultipleLocator, ScalarFormatter)

    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc)
    elif type(target).__base__ == plt.Axes:
        ax = target
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(loc)

    # Create levels, norms.
    levs = np.linspace(-1*zmax, zmax, 51)

    # Add contour.
    gradpr = data['gradpr'][data['z'] > 0]

    # Two contours: one for up, one for down.
    cn1 = ax.tricontourf(data['x'][data['z'] > 0], data['y'][data['z'] > 0],
                         gradpr, levs, cmap=plt.get_cmap('PuOr_r'),
                         extend='both')

    # Add colorbar.
    if add_cbar:
        fmt = ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_powerlimits((-4, 3))
        ticks = MultipleLocator(zmax/4)
        cb = plt.colorbar(cn1, ax=ax, shrink=.85, ticks=ticks, format=fmt)
        cb.set_label(r'$\nabla P_{Radial}$ ($nN/m^{3}$)')

    # Add latitude circles.
    rmax = data['r'][0]
    circ = Circle((0, 0), rmax, fill=False, ec='k', lw=1.5)
    ax.add_artist(circ)
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        r_text = r*np.sin(np.pi/4.0)
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0, ls='dashed')
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('GSM X ($R_E$)')
    if dolabely:
        ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*rmax, rmax])
    ax.set_ylim([-1.0*rmax, rmax])
    ax.hlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    if dolabel:
        ax.set_title('Shell at R=%4.2f$R_E$' % rmax)

    return fig, ax, cn1


def add_gradpr_plot_log(data, target=None, zmax=10, loc=111, dolabel=True,
                        add_cbar=False, dolabelx=True, dolabely=True):
    '''
    Create a dial plot of radial pressure gradient (force) and place it onto
    *target*, where *target* is either a figure, axes, or **None** (in which
    case a new figure will be generated.)
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator, LogFormatterMathtext

    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc)
    elif type(target).__base__ == plt.Axes:
        ax = target
        fig = ax.figure
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(loc)

    # Create levels, norms.
    tick = LogLocator()
    fmt = LogFormatterMathtext()
    zmin = 0.1
    lev1 = np.power(10, np.linspace(np.log10(zmin), np.log10(zmax), 50))

    # Add contour.
    up = data['gradpr'][data['z'] > 0]
    up[up < zmin] = zmin-zmin/10.0
    up[up > zmax] = zmax-zmax/10.0
    dn = data['gradpr'][data['z'] > 0]
    dn[dn > -0.01] = -0.001
    dn *= -1.

    # Two contours: one for up, one for down.
    cn1 = ax.tricontourf(data['x'][data['z'] > 0], data['y'][data['z'] > 0],
                         up, lev1, cmap=plt.get_cmap('Reds'), norm=LogNorm())
    cn2 = ax.tricontourf(data['x'][data['z'] > 0], data['y'][data['z'] > 0],
                         dn, lev1, cmap=plt.get_cmap('gray_r'), norm=LogNorm())

    # Add colorbar.
    if add_cbar:
        cb = plt.colorbar(cn1, ax=ax, shrink=.85, ticks=tick,
                          format=fmt, pad=-.07)
        cb.set_label(r'$\nabla P_{Radial}$ ($nN/m^{3}$)')
        cb = plt.colorbar(cn2, ax=ax, shrink=.85, ticks=tick, format='')

    # Add latitude circles.
    rmax = data['r'][0]
    circ = Circle((0, 0), rmax, fill=False, ec='k', lw=1.5)
    ax.add_artist(circ)
    for theta in range(30, 90, 15):
        r = rmax*np.cos(theta*np.pi/180.0)
        r_text = r*np.sin(np.pi/4.0)
        circ = Circle((0, 0), r, fill=False, ec='k', lw=1.0, ls='dashed')
        ax.add_artist(circ)
        ax.text(r_text, r_text, '%2i' % theta + r'$^{\circ}$')

    # Adjust axes properly.
    ax.set_aspect('equal')
    if dolabelx:
        ax.set_xlabel('GSM X ($R_E$)')
    if dolabely:
        ax.set_ylabel('GSM Y ($R_E$)')
    ax.set_xlim([-1.0*rmax, rmax])
    ax.set_ylim([-1.0*rmax, rmax])
    ax.hlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    ax.vlines(0, -1.*rmax, rmax, colors='k', linestyles='dotted')
    if dolabel:
        ax.set_title('Shell at R=%4.2f$R_E$' % rmax)

    return fig, ax, cn1, cn2


def calc_fluence(data, hemi='North', docheck=False):
    '''
    For a given shell of constant radius, integrate over the shell to get
    the total particle fluence in and out of the shell.  Defaults to
    northern hemisphere (e.g., GSM_Z > 0).
    '''

    import matplotlib.tri as triang

    # Start with some good constants:
    R = data['r'][0] * 6371.0  # radius in km.
    dTheta = 1.0 * np.pi/180.  # 1.0-degree spacing
    dPhi = 2.5 * np.pi/180.  # 2.5-degree spacing

    # Create grid according to selected hemisphere, limit
    # data use to proper hemisphere as well.  At same time, convert
    # 1/cm3 to 1/km3.
    Z = data['z']
    if hemi == 'North':
        theta = np.arange(0, np.pi/2.0, dTheta)
        phi = np.arange(0, 2.*np.pi,  dPhi)
        X, Y = data['x'][Z > 0], data['y'][Z > 0],
        FLUX = data['flux'][Z > 0] * 1000.**2  # 1/m -> 1/km
        # UR, N = data['ur'][Z>0], data['rho'][Z>0] * 1E15
    else:
        theta = np.arange(np.pi, np.pi/2.0, -1*dTheta)
        phi = np.arange(0, 2.*np.pi,  dPhi)
        X, Y = data['x'][Z < 0], data['y'][Z < 0]
        FLUX = data['flux'][Z < 0] * 1000.**2  # 1/m -> 1/km
        # UR, N = data['ur'][Z<0], data['rho'][Z<0] * 1E15

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
    tri = triang.Triangulation(X*6371.0, Y*6371.0)
    int_flux = triang.LinearTriInterpolator(tri, FLUX)  # , default_value=0.0)
    flux = int_flux(x, y)

    # Filter NaNs, which still happen sometime...
    flux_nans = flux[np.isnan(flux)].size + flux[np.isinf(flux)].size
    if flux_nans != 0:
        print("DANGER!  %i bad points found!" % flux_nans)
        flux[np.isnan(flux)] = 0
        flux[np.isinf(flux)] = 0

    # Save regularly gridded stuff into object:
    data['theta_reg'], data['phi_reg'] = theta, phi
    data['flux_reg'] = np.reshape(flux, (theta.size, phi.size))/1000.**2

    # Integrate!  Split by region, too.
    Flu = R**2 * flux * np.sin(theta_all) * dTheta * dPhi
    fluence = {'all': Flu.sum(),
               'noon': Flu[(phi_all > np.pi*7./4.) |
                           (phi_all < np.pi/4.)].sum(),
               'dusk': Flu[(phi_all > np.pi/4.) &
                           (phi_all < np.pi*3./4.)].sum(),
               'midn': Flu[(phi_all > np.pi*3./4.) &
                           (phi_all < np.pi*5./4.)].sum(),
               'dawn': Flu[(phi_all > np.pi*5./4.) &
                           (phi_all < np.pi*7./4.)].sum()}
    up = Flu[flux > 0]
    upphi = phi_all[flux > 0]
    upflu = {'all': up.sum(),
             'noon': up[(upphi > np.pi*7./4.) | (upphi < np.pi / 4.)].sum(),
             'dusk': up[(upphi > np.pi / 4.) & (upphi < np.pi*3./4.)].sum(),
             'midn': up[(upphi > np.pi*3./4.) & (upphi < np.pi*5./4.)].sum(),
             'dawn': up[(upphi > np.pi*5./4.) & (upphi < np.pi*7./4.)].sum()}
    down = Flu[flux < 0]
    dwnphi = phi_all[flux < 0]
    downflu = {'all': down.sum(),
               'noon': down[(dwnphi > np.pi*7./4.) |
                            (dwnphi < np.pi/4.)].sum(),
               'dusk': down[(dwnphi > np.pi/4.) &
                            (dwnphi < np.pi*3./4.)].sum(),
               'midn': down[(dwnphi > np.pi*3./4.) &
                            (dwnphi < np.pi*5./4.)].sum(),
               'dawn': down[(dwnphi > np.pi*5./4.) &
                            (dwnphi < np.pi*7./4.)].sum()}

    if docheck:
        # We can approximate the flux by using averages times volume.
        avg_all = flux.mean()*2.*np.pi*R**2
        avg_up = flux[u > 0].mean()*2.*np.pi*R**2
        avg_dwn = flux[u < 0].mean()*2.*np.pi*R**2

        print("ROUGH CHECK RESULTS:")
        print("Total,     Integrated vs. Avg = %12.5E  %12.5E" %
              (fluence['all'], avg_all))
        print("Upwards,   Integrated vs. Avg = %12.5E  %12.5E" %
              (upflu['all'], avg_up))
        print("Downwards, Integrated vs. Avg = %12.5E  %12.5E" %
              (downflu['all'], avg_dwn))
        print("Integrated area = %12.5E" %
              (np.sum(R**2*np.sin(theta_all)*dTheta*dPhi)))
        print("Actual area = %12.5E" % (2.*np.pi*R**2))
        print("Quandrant values:")
        print("\tNoon = %12.5E" % (fluence['noon']))
        print("\tDusk = %12.5E" % (fluence['dusk']))
        print("\tMidn = %12.5E" % (fluence['midn']))
        print("\tDawn = %12.5E" % (fluence['dawn']))
        summed = fluence['noon'] + fluence['dusk'] + \
            fluence['dawn'] + fluence['midn']
        print("\t...differs from total by %12.5E (%05.2f%%)" %
              (fluence['all'] - summed,
               (fluence['all']-summed)/fluence['all']))

    return fluence, upflu, downflu


def read_fluence(infile):
    '''
    Read a fluence ascii data file and return it as a dictionary.
    '''

    f = open(infile, 'r')

    # Skip first header; use 2nd header for data keys.
    f.readline()
    names = f.readline().split()

    # Slurp remainder of file to determine length.
    lines = f.readlines()
    nLines = len(lines)

    # Create data container.
    data = {}
    for n in names:
        data[n.lower()] = np.zeros(nLines)

    # Read and parse, baby.
    for i, l in enumerate(lines):
        for n, d in zip(names, l.split()):
            data[n][i] = d

    # Create total fluences.
    for n in ['total', 'noon', 'dusk', 'dawn', 'midn']:
        data[n] = data['up_'+n] + data['dn_'+n]

    # And that's it.
    return data
