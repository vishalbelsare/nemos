import matplotlib.pyplot as plt

import nemos as nmo
from jaxopt.projection import projection_non_negative
import numpy as np


class NonNegativeRidge(nmo.regularizer.Ridge):
    allowed_solvers = ["ProjectedGradient"]

    def __init__(
            self,
            solver_name: str = "ProjectedGradient",
            solver_kwargs=None,
            regularizer_strength=1.0,
    ):
        super().__init__(solver_name, solver_kwargs=solver_kwargs, regularizer_strength=regularizer_strength)

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


reg = NonNegativeRidge(solver_kwargs={"tol": 10**-12}, regularizer_strength=0.01)
obs = nmo.observation_models.PoissonObservations(inverse_link_function=lambda x: x)
glm = nmo.glm.GLM(regularizer=reg, observation_model=obs, )

T = 50000
bas1 = nmo.basis.RaisedCosineBasisLinear(10)
bas2 = nmo.basis.RaisedCosineBasisLinear(10)
x1, x2 = np.random.uniform(0, 1, size=(2, T))


X = (bas1 + bas2).evaluate(x1, x2)
weights = 10 * np.abs(np.random.normal(size=X.shape[1]))
rate = np.dot(X, weights)

count = np.random.poisson(rate)

glm.fit(X[:, np.newaxis], count[:, np.newaxis])

plt.scatter(glm.coef_.flatten(), weights.flatten())