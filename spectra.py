import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from random import random

SW = (0.0, 10.0)
LB = 0.1
NF = 1

def points(sw=SW, step=0.01):
    return np.arange(sw[0], sw[1] + step, step)

def lorentzian(x0, points, lb=LB):
    return np.apply_along_axis(lambda x:(0.5 * lb) / (pi * (((x - x0)**2)) + ((0.5 * lb)**2)), 0, points)

def peak(x0, coupling, points):
    if(len(coupling) == 0):
        return lorentzian(x0, points)
    if(len(coupling) == 1):
        return (lorentzian(x0 - coupling[0] / 2, points) + lorentzian(x0 + coupling[0] / 2, points)) / 2
    return (peak(x0 - coupling[0] / 2, coupling[1:], points) + peak(x0 + coupling[0] / 2, coupling[1:], points)) / 2
    
def spectrum(peak_list, points):
    if(len(peak_list) == 1):
        return peak(peak_list[0, 0], peak_list[0,1], points)
    return peak(peak_list[0, 0], peak_list[0,1], points) + spectrum(peak_list[1:], points)
    
def noisify(spectrum, nf=NF):
    F = np.vectorize(lambda x:x + ((0.5 - random()) * nf))
    return F(spectrum)

shift_points = points()    
peak_list = np.array([
    [1.5, [1,1]],
    [2, []],
    [4, [2]],
    [6, [2,2,4,5]]
    ], dtype=object)
    
spec = spectrum(peak_list, shift_points)
noisy_spec = noisify(spec)

plt.figure()
sns.lineplot(x=shift_points, y=noisy_spec)
plt.savefig('lorentzian.png')
