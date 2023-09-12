def eccentricity (RE, ra):
        ## ra and rp are apoapsis and periapsis, respectively
    return np.abs((RE/ra) - 1)
    
###

def nodal_precession(majorAxis, ecc, omega, incl):
    
        ## constants
    J2 = 1.08262668E-3   ## Earth's oblateness
    RE = 6378.137 * 1000.       ## in km
    
    return (-3./2.) * np.power(RE,2) / \
        np.power(majorAxis * (1 - np.power(ecc,2)), 2) * omega * \
        np.cos(incl) * J2

###

def apsidal_precession(majorAxis, ecc, omega, incl):
        
        ## constants
    J2 = 1.08262668E-3   ## Earth's oblateness
    RE = 6378.137 * 1000.       ## in km
    
    return (3./4.) * np.power(RE,2) / \
        np.power(majorAxis * (1 - np.power(ecc,2)), 2) * omega \
        * J2 * (5 * np.power(np.cos(incl),2) - 1) 

### ### ### MAIN BODY ### ### ###
    
import numpy as np

    ## constants
RE = 6378.137 * 1000.      ## in km

    ## user input
    
perigee = 350. * 1000.
apogee = 2550. * 1000.
inclination = 82.
inclDelta = 2.
inclRad = inclination * np.pi/180.
semiMajor = (apogee + perigee + 2 * RE) / 2. 

eccentr = eccentricity(RE, semiMajor)
#semiMinor = semiMajor ( 1 - eccentr )


    ## period
T = 2. * np.pi * np.sqrt(np.power(semiMajor,3) / (3.986004418 * 1E14))
omega = 2 * np.pi / T

print("Eccentricity and period:",eccentr, T/60.,"\n")
print("Nodal Precession\n")

    ## nodal precession

inclMin = 80.
inclMax = 84.
inclStep = 1.0
    
for x in np.arange(inclMin, inclMax, inclStep):
    inclination = x
    inclRad = x * np.pi/180.
    nodalPrec = nodal_precession(semiMajor, eccentr, omega, inclRad)
    print(inclination, nodalPrec * 180./np.pi*3600 * 24., " per day",
          nodalPrec * 180./np.pi*3600 * 24.*30., "per month", 
          nodalPrec * 180./np.pi*3600 * 24.*365.25, "per year")

print("\nApsidal Precession\n")
for x in np.arange(inclMin, inclMax, inclStep):
    inclination = x
    inclRad = x * np.pi/180.
    apsPrec = apsidal_precession(semiMajor, eccentr, omega, inclRad)
    print(inclination, apsPrec * 180./np.pi*3600 * 24., " per day",
          apsPrec * 180./np.pi*3600 * 24.*30., "per month",
          apsPrec * 180./np.pi*3600 * 24.*365.25, "per year")
