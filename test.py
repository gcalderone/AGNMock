import numpy as np
import matplotlib.pyplot as plt
import AGNModel as m

# --------------------------------------------------------------------
# Create an AGNModel object
agn = m.AGNModel()

# --------------------------------------------------------------------
# Plot intrinsic spectrum
plt.title("Intrinsic spectrum")
plt.xlabel("Wavelength [A] (rest frame)")
plt.ylabel("Luminosity [erg s^{-1}]")

wl = np.logspace(2, 7, 5000)
total = agn.spectrum(wl)
mm = np.amax(wl * total)
plt.ylim((mm / 1000, mm * 2))
p1 = plt.loglog(wl, wl * total)
p2 = plt.loglog(wl, wl * agn.disk.spectrum(wl))
p3 = plt.loglog(wl, wl * agn.host.spectrum(wl))
p4 = plt.loglog(wl, wl * agn.torus.spectrum(wl))
p5 = plt.loglog(wl, wl * agn.blr.spectrum(wl))
p6 = plt.loglog(wl, wl * agn.nlr.spectrum(wl))
plt.show()


# --------------------------------------------------------------------
# Plot observed spectrum
plt.title("Observed spectrum")
plt.xlabel("Wl [A] (observer frame)")
plt.ylabel("Flux [erg s^{-1} cm^{-2}]")

agn.observe(2., 30.)

wl = np.logspace(1, 8, 5000)
total = agn.spectrum(wl)
mm = np.amax(wl * total)
plt.ylim((mm / 1000, mm * 2))
p1 = plt.loglog(wl, wl * total)
p2 = plt.loglog(wl, wl * agn.disk.spectrum(wl))
p3 = plt.loglog(wl, wl * agn.host.spectrum(wl))
p4 = plt.loglog(wl, wl * agn.torus.spectrum(wl))
p5 = plt.loglog(wl, wl * agn.blr.spectrum(wl))
p6 = plt.loglog(wl, wl * agn.nlr.spectrum(wl))
plt.show()
