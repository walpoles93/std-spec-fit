import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from random import random

SW = (0.0, 10.0)
LB = 0.1
NF = 0.5

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
    
def std_peak(x0, coupling, points, std):
    return peak(x0, coupling, points) * std
    
def std_spectrum(peak_list, points, std_list):
    if(len(peak_list) == 1):
        return std_peak(peak_list[0, 0], peak_list[0,1], points, std_list[0])
    return std_peak(peak_list[0, 0], peak_list[0,1], points, std_list[0]) + std_spectrum(peak_list[1:], points, std_list[1:])    
    
def noisify(spectrum, nf=NF):
    F = np.vectorize(lambda x:x + ((0.5 - random()) * nf))
    return F(spectrum)
    
def solve_intensity(spectrum, peak_list, points):
    def F(x):
        predict = std_spectrum(peak_list, points, x) #TODO rename this function
        return np.sum((spectrum - predict)**2)
    return minimize(F, np.ones(len(peak_list)), bounds=Bounds(0, np.inf)).x

shift_points = points()    
peak_list = np.array([
    [1.5, [1,1]],
    [2, []],
    [4, [2]],
    [6, [2,2,4,5]]
    ], dtype=object)
std_list = [0.05, 0.1, 0.13, 0.02]
    
spec = spectrum(peak_list, shift_points)
noisy_spec = noisify(spec)

std_spec = std_spectrum(peak_list, shift_points, std_list)
noisy_std = noisify(std_spec)

spec_intensities = solve_intensity(noisy_spec, peak_list, shift_points)
std_intensities = solve_intensity(noisy_std, peak_list, shift_points)
print(std_intensities)
predicted_std_spec = std_spectrum(peak_list, shift_points, std_intensities)

plt.figure()
sns.lineplot(x=shift_points, y=noisy_spec)
sns.lineplot(x=shift_points, y=noisy_std)
sns.lineplot(x=shift_points, y=predicted_std_spec)
plt.savefig('lorentzian.png')
