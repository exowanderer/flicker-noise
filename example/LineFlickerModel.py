import sys
sys.path.append("../WaveletCode")
sys.path.append("../")
import FlickerLikelihood
import numpy as np
import pymc3 as pm
# from flicker import Wavelets

#  Functions  #


def model(t, a, b):
    return a * t + b


def flicker_likelihood(data, t, a, b, sigma_w, sigma_r):
    residuals = data - model(t, a, b)
    return FlickerLikelihood.get_likelihood(residuals, sigma_w, sigma_r)

# Get the data #
t, data = np.loadtxt('flicker_dataset.dat', unpack=True)

with pm.Model() as model:
    # Priors #
    a = pm.Uniform('a', -1, 1)
    b = pm.Uniform('b', -10, 10)
    sigma_w = pm.Uniform('sigma_w', 0, 100)
    sigma_r = pm.Uniform('sigma_r', 0, 100)

    # Likelihood #

    flicker_ll = flicker_likelihood(data, t, a, b, sigma_w, sigma_r)

    # Here we track the value of the model light curve for plotting purposes
    pm.Deterministic("light_curves", flicker_ll)

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=flicker_ll, sd=np.sqrt(abs(data)), observed=data)
