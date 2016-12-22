"""
Library for geometric computations. 
"""


import scipy.interpolate as interp
import numpy as np


"""
xs = np.arange(-0.99, 1, 0.001)
y0s = np.nan_to_num((1 - xs**2)**0.5)
y1s = np.nan_to_num(-xs / (1 - xs**2)**0.5)
y2s = np.nan_to_num(-1 / (1 - xs**2)**1.5)
ks = np.abs(y2s) / (1 + y1s**2)**1.5
libplot.plot(xs, np.transpose([ks, cs]), filepath='plot_semicircle.png')
"""

def cal_curvatures(f):
    pass


def est_curvatures(xs, ys, order=3):
    """
    Return unsigned curvatures. 
    
    Input: 
        order: order of interpolating polynomials
    """
    f = interp.InterpolatedUnivariateSpline(xs, ys, k=order)
    z1s, z2s = np.array([f.derivatives(x)[1:3] for x in xs]).transpose()
    ks = np.abs(z2s) / (1 + z1s**2)**1.5
    return ks

 