import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

from multiprocessing import cpu_count

from pymc3_optimize import optimize as pm_optimize
from pymc3_optimize import get_dense_nuts_step

size = 200
true_intercept = 1
true_slope = 2
tune = 5000
draws = 5000
target_accept = 0.9

with pm.Model() as model:
    # Model specifications in PyMC3 are wrapped in a with-statement

    # Define priors
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.Normal('Intercept', 0, sigma=20)
    x_coeff = pm.Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept + x_coeff * x,
                           sigma=sigma, observed=y)

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

with pm.Model() as model:
    # specify glm and pass in data. The resulting linear model,
    #   its likelihood and all its parameters are automatically
    #   added to our model.
    pm.glm.GLM.from_formula('y ~ x', data)

    map_soln1 = pm_optimize(start=model.test_point)

    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    # trace = pm.sample(3000, cores=2)

    trace1 = pm.sample(
        tune=tune,
        draws=draws,
        start=map_soln1,
        chains=cpu_count(),
        step=get_dense_nuts_step(target_accept=target_accept),
        cores=cpu_count()
    )

plt.ion()

n_burnin = 100
n_samples = 100
pm.traceplot(trace0[burnin:])
plt.tight_layout()


plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
pm.plot_posterior_predictive_glm(trace0, samples=n_samples,
                                 label='posterior predictive regression lines')
plt.plot(x, true_regression_line,
         label='true regression line', lw=3., c='y', alpha=0.2)

plt.title('Manual Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y')

# plt.figure(figsize=(7, 7))
pm.traceplot(trace1[n_burnin:])
plt.tight_layout()


plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
pm.plot_posterior_predictive_glm(trace1, samples=n_samples,
                                 label='GLM posterior predictions')
plt.plot(x, true_regression_line,
         label='true regression line', lw=3., c='y', alpha=0.2)

plt.title('GLM Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y')

plt.show()
"""
import LineFlickerModel

# Set the number of iterations and burnins:
niterations = 1e4
nburn = 1e3

# Set MAP estimates as starting point of MCMC. First, calculate MAP estimates:
M = pymc.MAP(LineFlickerModel)
M.fit()
# Now set this as starting point for the MCMC:
mc = pymc.MCMC(M.variables)

# And start the sample!
mc.sample(iter=niterations + nburn, burn=nburn)

# Plot the final samples (posterior samples):
pymc.Matplot.plot(mc)
plt.show()
"""
