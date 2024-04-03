r"""Example of constrained GLM with fixed intercept.

For positive $X$, this ProjectedGradientGLM solves,

$$
\text{argmin}_{\bm{w}} \mathcal{L}(\bm{w}|\; X, y)
$$

subject to,

$$
     \bm{x}_{\text{max}} \cdot \bm{w} \le - \frac{1}{T} \sum_t{y_t}.
$$

where  $\mathcal{L}$ is the Poisson log-likelihood with an identity link, and,

$$
\bm{x}_{\text{max}} = [\max_t(X\_{t, 1}), \dots, \max_t(X\_{t, n})]
$$

"""

from typing import Tuple

import jax.nn
import matplotlib.pyplot as plt
from jax import numpy as jnp

import nemos as nmo
from jaxopt.projection import projection_non_negative, projection_hyperplane
import numpy as np
import jaxopt

from nemos.base_class import DESIGN_INPUT_TYPE

jax.config.update("jax_enable_x64", True)
def mean_center(func):
    def wrapper(self, *args, **kwargs):
        if "X" in kwargs:
            kwargs["X"] = X - X.mean(axis=0)
        else:
            # assume X is first
            flat, struct = jax.tree_util.tree_flatten((args, kwargs))
            flat[0] = flat[0] - flat[0].mean(axis=0)
            args, kwargs = jax.tree_util.tree_unflatten(struct, flat)
        return func(self, *args, **kwargs)

    return wrapper

class ProjectedGradientReg(nmo.regularizer.UnRegularized):
    allowed_solvers = ["ProjectedGradient"]

    def __init__(
            self,
            solver_name: str = "ProjectedGradient",
            solver_kwargs=None
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)


class ProjectedGradientGLM(nmo.glm.GLM):
    r"""GLM with fixed intercept, positive inputs and identity link.

    This solves,

    $$
    \text{argmin}_{\bm{w}} \mathcal{L}(\bm{w}|\; X, y)
    $$

    subject to,

    $$
         \bm{x}_{\text{max}} \cdot \bm{w} \le - \frac{1}{T} \sum_t{y_t}.
    $$

    where  $\mathcal{L}$ is the Poisson log-likelihood with an identity link, and,

    $$
    \bm{x}_{\text{max}} = [\max_t(X\_{t, 1}), \dots, \max_t(X\_{t, n})]
    $$
    """
    def __init__(self,
                 observation_model=nmo.observation_models.PoissonObservations(inverse_link_function=lambda x: x),
                 regularizer=ProjectedGradientReg(solver_kwargs=dict(projection=projection_hyperplane, tol=10**-12)),
                 ):
        super().__init__(observation_model, regularizer)
        self._mean_rate = None

    @nmo.type_casting.support_pynapple(conv_type="jax")
    def _get_hyperplane_params(self, X, y):
        a = jax.tree_map(lambda x: -jnp.max(x, axis=0), X)
        b = jax.tree_map(lambda x: jnp.mean(x, axis=0, keepdims=True), y)

        # check that the rate is non-negative if the condition is matched
        assert nmo.utils.pytree_map_and_reduce(lambda x: jnp.all(jnp.dot(x, a) >= -b), all, X)

        return a, b

    def _check_params(self,params,data_type=None):
        return super()._check_params((params, self.intercept_))

    def _check_input_and_params_consistency(self, params, X=None, y=None):
        return super()._check_input_and_params_consistency((params, self.intercept_), X=X, y=y)

    @mean_center
    def fit(self, X, y, *args, init_params=None, **kwargs):
        # very important step, each feature should be mean centered, so that wi*xi has mean 0
        # and the mean of the counts corresponds to the mean of the counts

        #X = jax.tree_map(lambda x: x - x.mean(axis=0), X)
        # add the projection to the solver
        self.intercept_ = jax.tree_map(lambda x: jnp.mean(x, axis=0, keepdims=True), y)
        self.regularizer.solver_kwargs.update(dict(projection=projection_hyperplane))
        hyperparam = self._get_hyperplane_params(X, y)
        init_params = self._initialize_parameters(X, y)[0]
        return super().fit(X, y, hyperparam, *args, init_params=init_params, **kwargs)

    def _set_coef_and_intercept(self, params):
        self.coef_ = params

    def _get_coef_and_intercept(self):
        return self.coef_

    def _predict(
            self, params: DESIGN_INPUT_TYPE, X: jnp.ndarray
    ) -> jnp.ndarray:
        # ignore intercept
        return self._observation_model.inverse_link_function(
            # First, multiply each feature by its corresponding coefficient,
            # then sum across all features and add the intercept, before
            # passing to the inverse link function
            nmo.tree_utils.pytree_map_and_reduce(
                lambda w, x: jnp.einsum("k,tk->t", w, x), sum, params, X
            )
            + self.intercept_
        )

    @mean_center
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        return super().predict(X)

    @mean_center
    def score(self, X, y):
        return super().score(X, y)

# %%
# ## Simulate some data

T = 50000
bas1 = nmo.basis.RaisedCosineBasisLinear(10)
bas2 = nmo.basis.RaisedCosineBasisLinear(10)
x1, x2 = np.random.uniform(0, 1, size=(2, T))


X = (bas1 + bas2)(x1, x2)


X = X - X.mean(axis=0)


weights = 10 * np.random.normal(size=X.shape[1])
rate_modulator = np.dot(X, weights)
mean_rate = -rate_modulator.min() + 10
rate = rate_modulator + mean_rate
count = np.random.poisson(rate)

# %%
# ## Model Fit

model = ProjectedGradientGLM()
model.fit(X, count)

fig, axs = plt.subplots(1, 2)
axs[0].set_aspect("equal")
axs[0].set_title("weights")
axs[0].scatter(weights, model.coef_)
if np.diff(axs[0].get_xlim()) > np.diff(axs[0].get_ylim()):
    axs[0].set_ylim(axs[0].get_xlim())
else:
    axs[0].set_xlim(axs[0].get_ylim())
axs[0].set_xlabel("true")
axs[0].set_ylabel("model fit")
axs[0].plot(axs[0].get_xlim(),axs[0].get_xlim(),"k")

axs[1].set_aspect("equal")
axs[1].set_title("rate")
axs[1].scatter(rate[::100], model.predict(X)[::100])
if np.diff(axs[1].get_xlim()) > np.diff(axs[1].get_ylim()):
    axs[1].set_ylim(axs[1].get_xlim())
else:
    axs[1].set_xlim(axs[1].set_ylim())
axs[1].set_xlabel("true")
axs[1].set_ylabel("model predict")
axs[1].plot(axs[1].get_xlim(), axs[1].get_xlim(),"k")
plt.tight_layout()