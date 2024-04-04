import jaxopt
import nemos as nmo
import numpy as np
import matplotlib.pyplot as plt
import jax


class BatchSolver:
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def run(self, *args, **kwargs):
        #minibach
        return self._solver.run(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self._solver.step(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._solver = self._optimizer(*args, **kwargs)
        return self


T = 50000
bas1 = nmo.basis.RaisedCosineBasisLinear(10)
bas2 = nmo.basis.RaisedCosineBasisLinear(10)
x1, x2 = np.random.uniform(0, 1, size=(2, T))


X = (bas1 + bas2)(x1, x2)


weights = np.random.normal(size=X.shape[1])
rate = jax.nn.softplus(np.dot(X, weights) + 1)
count = np.random.poisson(rate)

reg = nmo.regularizer.UnRegularized(allow_custom_solvers=True, solver=BatchSolver(jaxopt.GradientDescent))
obs = nmo.observation_models.PoissonObservations(inverse_link_function=jax.nn.softplus)
model = nmo.glm.GLM(regularizer=reg, observation_model=obs)
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
