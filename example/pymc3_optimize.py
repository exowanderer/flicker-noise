# Taken from github.com/dfm/exoplanet/exoplanet/utils.py
#   We didn't want to make `exoplanet` as a requirement of flicker
import numpy as np
import pymc3 as pm

from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import Point
from pymc3.theanof import inputvars

from pymc3.util import (
    get_default_varnames,
    update_start_vals,
    get_untransformed_name,
    is_transformed_name,
)

from quadpotential import get_dense_nuts_step

import theano
import tqdm
import sys


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))


def get_theano_function_for_var(var, model=None, **kwargs):
    model = pm.modelcontext(model)
    kwargs["on_unused_input"] = kwargs.get("on_unused_input", "ignore")
    return theano.function(model.vars, var, **kwargs)


def get_args_for_theano_function(point=None, model=None):
    model = pm.modelcontext(model)
    if point is None:
        point = model.test_point
    return [point[k.name] for k in model.vars]


def optimize(
    start=None,
    vars=None,
    model=None,
    return_info=False,
    verbose=True,
    **kwargs
):
    """Maximize the log prob of a PyMC3 model using scipy

    All extra arguments are passed directly to the ``scipy.optimize.minimize``
    function.

    Args:
        start: The PyMC3 coordinate dictionary of the starting position
        vars: The variables to optimize
        model: The PyMC3 model
        return_info: Return both the coordinate dictionary and the result of
            ``scipy.optimize.minimize``
        verbose: Print the success flag and log probability to the screen

    """
    from scipy.optimize import minimize

    model = pm.modelcontext(model)

    # Work out the full starting coordinates
    if start is None:
        start = model.test_point
    else:
        update_start_vals(start, model.test_point, model)

    # Fit all the parameters by default
    if vars is None:
        vars = model.cont_vars
    vars = inputvars(vars)
    allinmodel(vars, model)

    # Work out the relevant bijection map
    start = Point(start, model=model)
    bij = DictToArrayBijection(ArrayOrdering(vars), start)

    # Pre-compile the theano model and gradient
    nlp = -model.logpt
    grad = theano.grad(nlp, vars, disconnected_inputs="ignore")
    func = get_theano_function_for_var([nlp] + grad, model=model)

    if verbose:
        names = [
            get_untransformed_name(v.name)
            if is_transformed_name(v.name)
            else v.name
            for v in vars
        ]
        sys.stderr.write(
            "optimizing logp for variables: [{0}]\n".format(", ".join(names))
        )
        bar = tqdm.tqdm()

    # This returns the objective function and its derivatives
    def objective(vec):
        res = func(*get_args_for_theano_function(bij.rmap(vec), model=model))
        d = dict(zip((v.name for v in vars), res[1:]))
        g = bij.map(d)
        if verbose:
            bar.set_postfix(logp="{0:e}".format(-res[0]))
            bar.update()
        return res[0], g

    # Optimize using scipy.optimize
    x0 = bij.map(start)
    initial = objective(x0)[0]
    kwargs["jac"] = True
    info = minimize(objective, x0, **kwargs)

    # Only accept the output if it is better than it was
    x = info.x if (np.isfinite(info.fun) and info.fun < initial) else x0

    # Coerce the output into the right format
    vars = get_default_varnames(model.unobserved_RVs, True)
    point = {
        var.name: value
        for var, value in zip(vars, model.fastfn(vars)(bij.rmap(x)))
    }

    if verbose:
        bar.close()
        sys.stderr.write("message: {0}\n".format(info.message))
        sys.stderr.write("logp: {0} -> {1}\n".format(-initial, -info.fun))
        if not np.isfinite(info.fun):
            logger.warning("final logp not finite, returning initial point")
            logger.warning(
                "this suggests that something is wrong with the model"
            )
            logger.debug("{0}".format(info))

    if return_info:
        return point, info
    return point
