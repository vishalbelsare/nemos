import jaxopt
import nemos as nmo
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_debug_nans", True)


def _batch_generator(batch_size, *args, **kwargs):
    """Yield mini-batches from the dataset."""
    num_samples = _get_n_samples(*args, **kwargs)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # Shuffle the data indices
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield _slice_args(batch_indices, *args, **kwargs)

def _slice_args(idx, *args, **kwargs):
    is_array = jax.tree_map(lambda x: isinstance(x, jax.numpy.ndarray), (args, kwargs))
    return jax.tree_map(lambda arr, bl: arr[idx] if bl else arr, (args, kwargs), is_array)


def _get_n_samples(*args, **kwargs):
    leaves = jax.tree_util.tree_leaves((args, kwargs))
    arrays = tuple(arr for arr in leaves if isinstance(arr, jax.numpy.ndarray))
    return arrays[0].shape[0]


class BatchSolver:
    def __init__(self, optimizer, batch_size=32, max_iter=1000):
        self._optimizer = optimizer
        self.batch_size = batch_size
        self._solver = None
        self._opt_state = None
        self._max_iter = max_iter

    def run(self, init_params, *args, **kwargs):
        # minibach
        opt_state = self._solver.init_state(init_params, *args, **kwargs)
        params = init_params
        for epoch in range(self._max_iter):
            for aa, kk in _batch_generator(self.batch_size, *args, **kwargs):
                params, opt_state = self.update(params, opt_state, *aa, **kk)
        return params, opt_state

    def update(self, params, state, *args, **kwargs):
        return self._solver.update(params, state, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._solver = self._optimizer(*args, **kwargs)
        return self

# Generate X and y
T = 5000
bas1 = nmo.basis.RaisedCosineBasisLinear(10)
bas2 = nmo.basis.RaisedCosineBasisLinear(10)
x1, x2 = np.random.uniform(0, 1, size=(2, T))
X = (bas1 + bas2)(x1, x2)
weights = np.random.normal(size=X.shape[1])
rate = jax.nn.softplus(np.dot(X, weights) + 1)
y = np.random.poisson(rate)

# fit no batching
model = nmo.glm.GLM()
model.fit(X, y)


# fit batching using solver directly
batch_size = 512
epochs = 100
step_size = 0.1
loss = model._predict_and_compute_loss
slv = BatchSolver(jaxopt.GradientDescent, batch_size=batch_size, max_iter=epochs)
slv = slv(loss)
par0 = np.zeros((X.shape[1])), np.log(y.mean(keepdims=True))
np.random.seed(123)
trained_params, state = slv.run(par0,jnp.asarray(X), jnp.asarray(y))


# fit batched using nemos
model_batch = nmo.glm.GLM(regularizer=
                    nmo.regularizer.UnRegularized(
                        allow_custom_solvers=True,
                        solver=BatchSolver(jaxopt.GradientDescent, batch_size=batch_size, max_iter=epochs)
                    )
)
np.random.seed(123)
model_batch.fit(X, y)



fig, axs = plt.subplots(1, 2)
axs[0].set_aspect("equal")
axs[0].set_title("weights")
axs[0].scatter(weights, model.coef_)
axs[0].scatter(weights, trained_params[0], color="r", s=80)
axs[0].scatter(weights, model_batch.coef_, color="k")
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


