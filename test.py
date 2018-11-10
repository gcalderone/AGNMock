import numpy as np
import matplotlib.pyplot as plt
import AGNModel as m

# --------------------------------------------------------------------
# Create an AGNModel object
agn = m.AGNModel()

# --------------------------------------------------------------------
# Plot intrinsic spectrum
wavelength = np.logspace(2, 7, 5000)
spec = agn.intrinsic_spectrum(wavelength)
mm = np.amax(wavelength * spec)
plt.ylim((mm / 1000, mm * 2))
plt.title("Intrinsic spectrum")
plt.xlabel("Wavelength [A] (rest frame)")
plt.ylabel("Luminosity [erg s^{-1}]")
p1 = plt.loglog(wavelength, wavelength * spec, label='Intrinsic spectrum')

# Plot individual components
p2 = plt.loglog(wavelength, wavelength * agn.norm_disk  * agn.disk.intrinsic_spectrum(wavelength))
p3 = plt.loglog(wavelength, wavelength * agn.norm_host  * agn.host.intrinsic_spectrum(wavelength))
p4 = plt.loglog(wavelength, wavelength * agn.norm_torus * agn.torus.intrinsic_spectrum(wavelength))
p5 = plt.loglog(wavelength, wavelength * agn.norm_blr   * agn.blr.intrinsic_spectrum(wavelength))
p6 = plt.loglog(wavelength, wavelength * agn.norm_blr   * agn.nlr.intrinsic_spectrum(wavelength))
plt.legend(handles=[p1, p2, p3, p4, p5, p6])
plt.show()


# --------------------------------------------------------------------
# Plot observed spectrum
agn.redshift = 2
agn.viewangle = 30
wavelength = np.logspace(1, 8, 5000)
flux = agn.observed_spectrum(wavelength)
mm = np.amax(wavelength * flux)
plt.ylim((mm / 1000, mm * 2))
plt.title("Observed spectrum")
plt.xlabel("Wavelength [A] (observer frame)")
plt.ylabel("Flux [erg s^{-1} cm^{-2}]")
plt.loglog(wavelength, wavelength * flux)
plt.show()




