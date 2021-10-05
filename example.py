import numpy as np
import matplotlib.pyplot as plt
import AGNMock as m

# --------------------------------------------------------------------
# Create an AGNMock object
agn = m.AGNMock()


# --------------------------------------------------------------------
# Set model properties

# BH mass
agn.disk.bh_logmass = 8.         # log10(M / M_sun)

# Accretion disk inner radius (used to calculate radiative efficiency
agn.disk.rin = 6.                # gravitational radius: 6:Schwarzschild, 1:Maximally rotating, 9:Maximally counter-rotating

# Disk emitted bolometric luminosity can be specified using one of the
# following methods (the other quantities will be automatically
# re-calculated):
agn.disk.eddington_ratio = 0.02  # must be in the range (0,1)
agn.disk.accrate = 0.06          # accretion rate, M_sun yr^-1
agn.disk.luminosity = 3e44       # erg / s

# Broad line region
agn.blr.luminosity = 1.e42       # Luminosity of Ly-alpha [erg s^-1] (the remaining lines are scaled accordingly)
agn.blr.fwhm  = 5e3              # FWHM of all lines in BLR [km s^-1]

# Narrow line region
agn.nlr.luminosity = 1.5e41      # Luminosity of Ly-alpha [erg s^-1] (the remaining lines are scaled accordingly)
agn.nlr.fwhm  = 5e2              # FWHM of all lines in NLR [km s^-1]

# Host galaxy
agn.host.template = 'Ell5'       # Choose one from the SWIRE collection (http://www.iasf-milano.inaf.it/~polletta/templates/swire_templates.html)
agn.host.luminosity = 5e43       # νL_ν luminosity @ 5500Å

# Torus
agn.torus.luminosity = 1e44      # total integrated luminosity [erg s^-1], typically ~1/3 of the disk bolometric luminosity


# The model already comes with pre-defined values when it is created.
# Such values can either be changed (as shown above) or left at their
# default values.  The latter can be restored at any time with:
#
# agn.defaultValues()



# --------------------------------------------------------------------
# Plot intrinsic (emitted) spectrum
plt.title("Intrinsic spectrum")
plt.xlabel("Wavelength [A] (rest frame)")
plt.ylabel("λL_λ luminosity [erg s^{-1}]")

# Prepare a grid of wavelengths to evaluate the model
wl = np.logspace(2, 7, 5000)

total = agn.spectrum(wl)
mm = np.amax(wl * total)
plt.ylim((mm / 1000, mm * 2))
p2, = plt.loglog(wl, wl * agn.disk.spectrum(wl), label="Disk")
p3, = plt.loglog(wl, wl * agn.host.spectrum(wl), label="Host galaxy")
p4, = plt.loglog(wl, wl * agn.torus.spectrum(wl), label="Torus")
p1, = plt.loglog(wl, wl * total, label="Total")
plt.legend(handles=[p1,p2,p3,p4])
plt.show()


# --------------------------------------------------------------------
# Plot observed spectrum
plt.title("Observed spectrum")
plt.xlabel("Wavelength [A] (observer frame)")
plt.ylabel("λF_λ flux [erg s^{-1} cm^{-2}]")

# Place the source at z=2 with a disk inclination of 30 degrees w.r.t
# the line of sight (a pole on line of sight corresponds to 0 degrees)
agn.observe(2., 30.)

# Prepare a grid of wavelengths to evaluate the model
wl = np.logspace(1, 8, 5000)

total = agn.spectrum(wl)
mm = np.amax(wl * total)
plt.ylim((mm / 1000, mm * 2))
plt.xlim(500, 2e6)
plt.ylim(1e-15, 2e-14)

p2, = plt.loglog(wl, wl * agn.disk.spectrum(wl), label="Disk")
p3, = plt.loglog(wl, wl * agn.host.spectrum(wl), label="Host galaxy")
p4, = plt.loglog(wl, wl * agn.torus.spectrum(wl), label="Torus")
p1, = plt.loglog(wl, wl * total, label="Total")
plt.legend(handles=[p1,p2,p3,p4])
plt.show()
