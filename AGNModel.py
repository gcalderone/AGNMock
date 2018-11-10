import math
import numpy as np
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM

class PhysicalConstants:
    def __init__(self):
        self.g = 6.6725899594475280e-08
        self.c = 29979245800.
        self.sunmass = 1.99e+33
        self.year = 31536000.
        self.eddington = 1.26e+38
        self.sigma = 5.6705098567508688e-05
        self.h = 6.6260756805287591e-27
        self.k = 1.3806580232811846e-16
        self.pc = 3.0859999515706286e+18


class AGNComponent:
    def __init__(self):
        self._name = ""
        self._redshift = 0.
        self._viewangle = 0.
        self._f2l = 1.

    def _observe(self, z, v, f2l):
        self._redshift = z
        self._viewangle = v
        self._f2l = f2l

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def spectrum(self, args):
        print("Error: this method must be implemented in derived classes")


class AccretionDisk(AGNComponent):
    def __init__(self):
        AGNComponent.__init__(self)
        self.defaultValues()

    def defaultValues(self):
        self._bhmass = 1.e8
        self._accrate = 0.1
        self._rin = 6.
        self._rout = 1000.
        self.luminosity = 3.e44

    @property
    def bh_logmass(self): # [log M_sun]
        return math.log10(self._bhmass)

    @bh_logmass.setter
    def bh_logmass(self, value):
        if value < 0:
            raise ValueError("bh_logmass must be a positive number")
        self._bhmass = 10.**value

    def grav_radius_cm(self):
        p = PhysicalConstants()
        return p.g * self._bhmass * p.sunmass / p.c**2

    def grav_radius_light_hours(self):
        p = PhysicalConstants()
        return self.grav_radius_cm / (p.c * 3600.)

    @property
    def rin(self): # [R_g, 6:Schwarzschild, 1:Maximally rotating, 9:Maximally counter-rotating]
        return self._rin

    @rin.setter
    def rin(self, value):
        if not (1 <= value < 9):
            raise ValueError("Rin must be in the range 1..9")
        self._rin = value

    @property
    def rout(self): # [R_g]
        return self._rout

    @rout.setter
    def rout(self, value):
        if value <= 9:
            raise ValueError("Rout must be > 9")
        self._rout = value

    def radiative_efficiency(self):
        return (1. / self._rin - 3. / self._rout) / 2.

    @property
    def accrate(self): # [M_sun yr^-1]
        return self._accrate

    @accrate.setter
    def accrate(self, value):
        if value < 0:
            raise ValueError("Accrate must be a positive number")
        self._accrate = value

    @property
    def luminosity(self):  # [erg s^-1]
        p = PhysicalConstants()
        eta = self.radiative_efficiency()
        return self._accrate * eta * p.sunmass / p.year * p.c**2

    @luminosity.setter
    def luminosity(self, value):
        if value < 0:
            raise ValueError("Luminosity must be a positive number")
        p = PhysicalConstants()
        eta = self.radiative_efficiency()
        self._accrate = value / eta / p.sunmass * p.year / p.c**2

    @property
    def eddington_ratio(self):
        p = PhysicalConstants()
        return self.luminosity / self._bhmass / p.eddington

    @eddington_ratio.setter
    def eddington_ratio(self, value):
        if value < 0:
            raise ValueError("Eddington_Ratio must be a positive number")
        p = PhysicalConstants()
        self.luminosity = value * self._bhmass * p.eddington

    def spectrum(self, wavelength):
        restlambda = wavelength / (1. + self._redshift)
        p = PhysicalConstants()
        Mass = self._bhmass * p.sunmass # BH mass in g

        # Compute accretion rate in g s^-1
        Mdot = self._accrate * p.sunmass / p.year

        # Compute number of radial points so that in standard
        # configuration (rin = 6, r2 = 1000) we have 10000 points.
        nrad = int(round(math.log10(self._rout / self._rin) / math.log10(1000./6) * 10000))

        # Radius [cm]
        rin  = self._rin  * self.grav_radius_cm()
        rout = self._rout * self.grav_radius_cm()
        r = np.logspace(math.log10(rin), math.log10(rout), nrad)
        r = r[1:] # Neglect Rin since it leads to a division by 0

        # Flux emitted from each annulus (Eq. A1)
        flux = 3 * p.g * Mdot * Mass
        flux /= (8 * 3.1416 * r**3)
        flux *= (1 - np.sqrt(rin/r))

        #Temperature [K]
        t = (flux / p.sigma)**(1./4)

        # Compute spectrum (luminosity density, Eq. A7)
        spec = np.arange(restlambda.size, dtype=float)
        for i in range(restlambda.size):
            freq = p.c * 1.e8 / restlambda[i]
            bb = 2. * p.h * freq**3 / p.c**2 / (np.exp(p.h * freq / (p.k * t)) - 1)
            spec[i] = 4. * 3.1416**2. * simps(r * bb, r)
        spec *= p.c * 1.e8 / restlambda / restlambda

        if self._f2l > 0:
            spec *= self._f2l * 2. * math.cos(self._viewangle * 3.1416/180)
        return spec


# ====================================================================
class EmissionLine:
    def __init__(self, label, wavelength):
        self._label = label
        self._lum = 1.e42
        self._wavelength = float(wavelength)
        self._fwhm = 3000.
        self._voff = 0.

    @property
    def label(self):
        return self._label

    @property
    def luminosity(self):  # [erg s^-1]
        return self._lum

    @luminosity.setter
    def luminosity(self, value):
        if value < 0:
            raise ValueError("Luminosity must be a positive number")
        self._lum = value

    @property
    def ref_wavelength(self):  # [AA]
        return self._wavelength

    @property
    def fwhm(self):  # [km s^-1]
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        if value < 0:
            raise ValueError("fwhm must be a positive number")
        self._fwhm = value

    @property
    def voff(self):  # [km s^-1]
        return self._voff

    @voff.setter
    def voff(self, value):
        self._voff = value

    @property
    def wavelength(self):  # [AA]
        p = PhysicalConstants()
        return (1. + self._voff * 1.e5 / p.c) * self._wavelength


class EmissionLineGaussian(EmissionLine):
    def __init__(self, label, wavelength):
        EmissionLine.__init__(self, label, wavelength)

    def spectrum(self, restlambda):
        p = PhysicalConstants()
        sigma = self.fwhm * 1.e5 / p.c * self.wavelength / 2.35
        ee = ((restlambda - self.wavelength) / sigma)**2. / 2.
        line = self.luminosity * np.exp( -ee ) / 2.50663 / sigma #SQRT(2*!PI) = 2.50663
        return line


# ====================================================================
class LineRegion(AGNComponent):
    def _addLine(self, label, wavelength, norm):
        tmp = EmissionLineGaussian(label, wavelength)
        tmp.luminosity = norm
        self._lines.append(tmp)

    def __init__(self):
        AGNComponent.__init__(self)
        self._lines  = []
        self.fwhm = 5000.
        self.voff = 0.
        self.luminosity = 1.e44

    @property
    def luminosity(self):  # [erg s^-1]
        return self._lum

    @luminosity.setter
    def luminosity(self, value):
        if value < 0:
            raise ValueError("Luminosity must be a positive number")
        self._lum = value

    @property
    def fwhm(self):  # [km s^-1]
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        if value < 0:
            raise ValueError("Fwhm must be a positive number")
        self._fwhm = value

    @property
    def voff(self):  # [km s^-1]
        return self._voff

    @voff.setter
    def voff(self, value):
        if value < 0:
            raise ValueError("Voff must be a positive number")
        self._voff = value

    def spectrum(self, wavelength):
        restlambda = wavelength / (1. + self._redshift)
        spec = np.arange(restlambda.size, dtype=float)
        spec *= 0.
        for line in self._lines:
            line.fwhm = self._fwhm
            line.voff = self._voff
            spec += line.spectrum(restlambda)
        spec *= self._f2l * self._lum 
        return spec


# ====================================================================
class BroadLineRegion(LineRegion):
    def defaultValues(self):
        self._addLine("Lya"       , 1215.24  , 1.)     # BN
        self._addLine("NV_1241"   , 1240.81  , 0.5)    # B
        self._addLine("OI_1306"   , 1305.53  , 0.035)  # B
        self._addLine("CII_1335"  , 1335.31  , 0.025)  # B
        self._addLine("SiIV_1400" , 1399.8   , 0.19)   # B
        self._addLine("CIV_1549"  , 1549.48  , 0.63)   # B
        self._addLine("CIII_1909" , 1908.734 , 0.29)   # B
        self._addLine("MgII_2798" , 2799.117 , 0.34)   # B
        self._addLine("Hd"        , 4102.89  , 0.028)  # B
        self._addLine("Hg"        , 4341.68  , 0.13)   # B
        self._addLine("Hb"        , 4862.68  , 0.22)   # BN
        self._addLine("HeI_5876"  , 5877.30  , 0.01)   # B
        self._addLine("Ha"        , 6564.61  , 0.40)   # BN

    def __init__(self):
        LineRegion.__init__(self)
        self.defaultValues()
        self.fwhm = 5000.
        self.luminosity = 1.e42


# ====================================================================
class NarrowLineRegion(LineRegion):
    def defaultValues(self):
        self._addLine("Lya"       , 1215.24  , 1.)    # BN
        self._addLine("NeVI_3426" , 3426.85  , 0.01)  # N
        self._addLine("OII_3727"  , 3729.875 , 0.0078)# N
        self._addLine("NeIII_3869", 3869.81  , 0.036) # N
        self._addLine("Hb"        , 4862.68  , 0.1)   # BN
        self._addLine("OIII_4959" , 4960.295 , 0.0093)# N
        self._addLine("OIII_5007" , 5008.240 , 0.034) # N
        self._addLine("NII_6549"  , 6549.86  , 0.01)  # N
        self._addLine("Ha"        , 6564.61  , 0.20)  # BN
        self._addLine("NII_6583"  , 6585.27  , 0.10)  # N
        self._addLine("SII_6716"  , 6718.29  , 0.10)  # N
        self._addLine("SII_6731"  , 6732.67  , 0.10)  # N

    def __init__(self):
        LineRegion.__init__(self)
        self.defaultValues()
        self.fwhm = 500.
        self.luminosity = 1.e42


# ====================================================================
class HostGalaxy(AGNComponent):
    def defaultValues(self):
        self._lum = 5.e43
        self._template = "Ell5"

    def __init__(self):
        AGNComponent.__init__(self)
        self.defaultValues()

    @property
    def luminosity(self):
        return self._lum

    @luminosity.setter
    def luminosity(self, value):
        if value < 0:
            raise ValueError("Luminosity must be a positive number")
        self._lum = value

    @property
    def template(self):
        return self._template

    @template.setter
    def template(self, value):
        self._template = value

    def spectrum(self, wavelength):
        restlambda = wavelength / (1. + self._redshift)
        fileName = "swire/" + self._template + "_template_norm.sed"
        x = np.arange(0, dtype=float)
        y = np.arange(0, dtype=float)
        for line in open(fileName, 'r'):
            a, b = line.split()
            x = np.append(x, float(a))
            y = np.append(y, float(b))
        y /= np.interp(5500., x, y) * 5500.
        y *= self._lum
        a = np.interp(restlambda, x, y)
        return self._f2l * a


# ====================================================================
class Torus(AGNComponent):
    def defaultValues(self):
        self._lum = 1.e44
        self._Tin  = 1500.
        self._Tout = 300.

    def __init__(self):
        AGNComponent.__init__(self)
        self.defaultValues()

    @property
    def luminosity(self):
        return self._lum

    @luminosity.setter
    def luminosity(self, value):
        if value < 0:
            raise ValueError("Luminosity must be a positive number")
        self._lum = value

    @property
    def Tin(self):
        return self._Tin

    @Tin.setter
    def Tin(self, value):
        if value < 0:
            raise ValueError("Tin must be a positive number")
        self._Tout = value

    @property
    def Tout(self):
        return self._Tout

    @Tout.setter
    def Tout(self, value):
        if value < 0:
            raise ValueError("Tout must be a positive number")
        self._Tout = value

    def spectrum(self, wavelength):
        restlambda = wavelength / (1. + self._redshift)
        ll = restlambda * 1.e-8
        p = PhysicalConstants()
        out1 = 1. / ll**5. / (np.exp(p.h * p.c / (ll * p.k * self._Tin )) - 1.)
        out2 = 1. / ll**5. / (np.exp(p.h * p.c / (ll * p.k * self._Tout)) - 1.)
        out1 /= simps(out1, restlambda)
        out2 /= simps(out2, restlambda)
        return self._f2l * self._lum * (out1+out2)


# ====================================================================
class AGNModel:
    def defaultValues(self):
        self.disk = AccretionDisk()
        self.host = HostGalaxy()
        self.torus = Torus()
        self.blr = BroadLineRegion()
        self.nlr = NarrowLineRegion()

    def __init__(self):
        self.defaultValues()

    def spectrum(self, wavelength):
        c1 = self.disk.spectrum(wavelength)
        c2 = self.host.spectrum(wavelength)
        c3 = self.torus.spectrum(wavelength)
        c4 = self.blr.spectrum(wavelength)
        c5 = self.nlr.spectrum(wavelength)
        return (c1 + c2 + c3 + c4 + c5)

    def observe(self, z=2., v=30.):
        if z <= 0:
            raise ValueError("Redshift must be a positive number")
        if (v < 0)  |  (v > 90):
            raise ValueError("Viewing angle must be in the range 0:90")

        p = PhysicalConstants()
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        Dl = cosmo.luminosity_distance(z)
        Dl = Dl.value * 1.e6 * p.pc
        f2l = 1 / (4. * 3.1416 * Dl**2.) / (1. + z)

        self.disk._observe(z, v, f2l)
        self.host._observe(z, v, f2l)
        self.torus._observe(z, v, f2l)
        self.blr._observe(z, v, f2l)
        self.nlr._observe(z, v, f2l)
