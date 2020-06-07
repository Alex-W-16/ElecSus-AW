'''File from which the goal functions for optimization are called'''

from __future__ import (division, print_function, absolute_import)

import scipy.integrate as integrate
import numpy as np


def FWHM(Detuning, Transmission):
    argOfMax = Transmission.argmax()
    heightOfMax = Transmission.max()

    HM = heightOfMax / 2.

    detuningLeft = Detuning[0:argOfMax]
    specLeft = Transmission[0:argOfMax]
    halfMaskBoolean = specLeft<HM
    hwLeft = detuningLeft[halfMaskBoolean][-1]

    detuningRight = Detuning[argOfMax:]
    specRight = Transmission[argOfMax:]
    halfMaskBoolean = specRight<HM
    hwRight = detuningRight[halfMaskBoolean][0]

    fwhm = hwRight - hwLeft

    return fwhm


def ENBW(Detuning, Transmission):
    #ENBW = integrate.simps(Transmission, Detuning) / Transmission.max()
    ENBW = integrate.trapz(Transmission, Detuning) / Transmission.max()
    #print(integrate.trapz(Transmission, Detuning)/Transmission.max())
    #print(ENBW)
    return ENBW


def FOM(Detuning, Transmission):
    FOM = Transmission.max() / ENBW(Detuning, Transmission)
    if np.isnan(FOM):
        FOM = 0
    return FOM.real

def targetENBW(Detuning, Transmission, targetTransmission):
    ENBW = integrate.simps(Transmission, Detuning) / targetTransmission
    return ENBW

def targetFOM(Detuning, Transmission, targetDetuning):
    index, = np.where(Detuning == targetDetuning)
    targetTransmission = np.asscalar(Transmission[index])
    FOM = targetTransmission / targetENBW(Detuning, Transmission, targetTransmission)
    if np.isnan(FOM):
        FOM = 0
    return FOM.real