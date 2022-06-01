'''
author: David S. N. Rupke
created: 2021dec02
modified:
notes:
'''
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter

class ExpModel(Fittable1DModel):
    '''
    Exponential model class for astropy.modeling package.

    y = yoff + amp*e^(-invefold*(x-xoff))

    Parameters
    ----------
        yoff (float)
        amp (float)
        xoff (float)
        invefold (float)

    Methods
    -------
        evaluate(x, yoff, amp, xoff, invefold)
        fit_deriv(x, yoff, amp, xoff, invefold)

    '''
    yoff = Parameter()
    amp = Parameter()
    xoff = Parameter()
    invefold = Parameter()
    linear = False

    @staticmethod
    def evaluate(x, yoff, amp, xoff, invefold):
        return yoff + amp*np.exp(-invefold*(x - xoff))

    @staticmethod
    def fit_deriv(x, yoff, amp, xoff, invefold):
        d_yoff = np.ones_like(x)
        d_amp = np.exp(-invefold*(x - xoff))
        d_xoff = amp*np.exp(-invefold*(x - xoff))*invefold
        d_invefold = amp*np.exp(-invefold*(x - xoff))*(xoff - x)
        return [d_yoff, d_amp, d_xoff, d_invefold]
