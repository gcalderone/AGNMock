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
plt.ylabel("Flux [erg s^{-1} cm^{-2}]")

agn.observe(2., 30.)

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
