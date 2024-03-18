from typing import Tuple

import matplotlib.pyplot as plt
from jax import numpy as jnp

import nemos as nmo
from jaxopt.projection import projection_non_negative
import numpy as np

from nemos.base_class import DESIGN_INPUT_TYPE


# Approach 1:
# Define a regularizer accepting constrained optimization.
class NonNegativeRidge(nmo.regularizer.UnRegularized):
    allowed_solvers = ["ProjectedGradient"]

    def __init__(
            self,
            solver_name: str = "ProjectedGradient",
            solver_kwargs=None
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs)

    def instantiate_solver(
        self, loss, *args, **kwargs
    ):
        """
        Instantiate the solver with a penalized loss function.

        Parameters
        ----------
        loss :
            The original loss function to be optimized.

        Returns
        -------
        Callable
            A function that runs the solver with the penalized loss.
        """
        self.solver_kwargs["projection"] = projection_non_negative
        return super().instantiate_solver(loss, None, *args, **kwargs)


# Approach 2: define a GLM like object
class PositiveWeightsLNP(nmo.glm.GLM):
    def _predict(
        self, params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], X: jnp.ndarray
    ) -> jnp.ndarray:
        params = (jnp.exp(params[0]), params[1])
        return super()._predict(params, X)


# GENERATE DATA
T = 50000
bas1 = nmo.basis.RaisedCosineBasisLinear(10)
bas2 = nmo.basis.RaisedCosineBasisLinear(10)
x1, x2 = np.random.uniform(0, 1, size=(2, T))


X = (bas1 + bas2).evaluate(x1, x2)
weights = 10 * np.abs(np.random.normal(size=X.shape[1]))
rate = np.dot(X, weights)
count = np.random.poisson(rate)

# APPROACH 1
reg = NonNegativeRidge(solver_kwargs={"tol": 10**-12} )
obs = nmo.observation_models.PoissonObservations(inverse_link_function=lambda x: x)
glm = nmo.glm.GLM(regularizer=reg, observation_model=obs, )
glm.fit(X[:, np.newaxis], count[:, np.newaxis])
plt.figure()
plt.title("Constrained optim")
plt.scatter(glm.coef_.flatten(), weights.flatten())
plt.xlabel("fit coef")
plt.ylabel("true coef")


# APPROACH 2
reg = nmo.regularizer.UnRegularized(solver_kwargs={"tol": 10**-12} )
obs = nmo.observation_models.PoissonObservations(inverse_link_function=lambda x: x)
glm = PositiveWeightsLNP(regularizer=reg, observation_model=obs, )
glm.fit(X[:, np.newaxis], count[:, np.newaxis])
plt.figure()
plt.title("log-weights")
plt.scatter(np.exp(glm.coef_.flatten()), weights.flatten())
plt.xlabel("fit coef")
plt.ylabel("true coef")



