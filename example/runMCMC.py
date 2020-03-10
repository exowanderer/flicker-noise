import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

from multiprocessing import cpu_count

from pymc3_optimize import optimize as pm_optimize
from pymc3_optimize import get_dense_nuts_step

# import LineFlickerModel
import sys


def straight_line(t, a, b):
    return a * t + b


def flicker_likelihood(data, t, a, b, sigma_w, sigma_r, n_data):
    residuals = data - straight_line(t, a, b)
    return get_likelihood(residuals, sigma_w, sigma_r, n_data)

if __name__ == '__main__':
    from flicker.FlickerLikelihood import get_likelihood

    # Get the data #
    t, data = np.loadtxt('flicker_dataset.dat', unpack=True)
    n_data = len(data)

    # Set the number of iterations and burnins:
    tune = 5000
    draws = 5000
    target_accept = 0.9

    with pm.Model() as model:
        # Priors #
        a = pm.Uniform('a', -1, 1)
        b = pm.Uniform('b', -10, 10)
        sigma_w = pm.Uniform('sigma_w', 0, 100)
        sigma_r = pm.Uniform('sigma_r', 0, 100)

        # Likelihood #

        flicker_ll = flicker_likelihood(
            data, t, a, b, sigma_w, sigma_r, n_data)

        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", flicker_ll)

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=flicker_ll, sd=np.sqrt(abs(data)), observed=data)

        # Model specifications in PyMC3 are wrapped in a with-statement
        map_soln0 = pm_optimize(start=model.test_point)

        # Inference!
        # Draw 5000 posterior samples using NUTS samplin
        trace0 = pm.sample(
            tune=tune,
            draws=draws,
            start=map_soln0,
            chains=cpu_count(),
            step=get_dense_nuts_step(target_accept=target_accept),
            cores=cpu_count()
        )

    n_burnin = 100
    n_samples = 100
    pm.traceplot(trace0[burnin:])
    plt.tight_layout()

    plt.figure(figsize=(7, 7))
    plt.plot(t, data, 'x', label='data')
    pm.plot_posterior_predictive_glm(trace0, samples=n_samples,
                                     label='posterior predictive regression lines')
    # plt.plot(x, true_regression_line,
    #          label='true regression line', lw=3., c='y', alpha=0.2)

    plt.title('Manual Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')

    # plt.figure(figsize=(7, 7))
    pm.traceplot(trace1[n_burnin:])
    plt.tight_layout()

    plt.show()
