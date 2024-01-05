# %%
# Import and load
import numpy as np
import scipy.io as sio


import matplotlib.pylab as plt
import nemos as nmo
import jax.numpy as jnp

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge

dat = sio.loadmat("/Users/ebalzani/Code/group_lasso_grant/group-lasso-glm/data/m691l1_xytNoise_stimulus.mat")
dat_activity = sio.loadmat("/Users/ebalzani/Code/group_lasso_grant/group-lasso-glm/data/m691l1#12_second_64_workspace.mat")

# %%
# Plot some of the stimuli
frame_tensor = dat['stim']['frames'][0, 0]
fig, axs = plt.subplots(1, 4)
frame0 = 1000
for k in range(4):
    axs[k].imshow(frame_tensor[..., frame0 + k])

# %%
# Create a time vector
dt_sec = 0.001
refresh_rate = dat['stim']['refresh_rate'][0, 0][0, 0]
time_sec = np.arange(frame_tensor.shape[2]) / refresh_rate
# edges for binning spikes
time_dt = np.arange(0, time_sec[-1] / dt_sec + 1) * dt_sec

frame_id = np.array(np.floor(time_dt * refresh_rate), dtype=int)

# %%
# Bin the spikes

# curated units. shape (n_curated_units, )
curated_units = dat_activity['clus'][0, :]
# all units. shape (n_units, )
unit_ids = dat_activity['xml']['spiketimesms4'][0, 0]['IDs'][0, 0][0, :]
# spike times, shape (n_units, )
spk_times = dat_activity['xml']['spiketimesms4'][0, 0]['Times'][0, 0][:, 0]

binned_spikes = np.zeros((time_dt.shape[0] - 1, curated_units.shape[0]))
cc = 0
for unt_id in curated_units:
    sel_unt = np.where(unit_ids == unt_id)[0]
    assert (len(sel_unt) == 1)
    sel_unt = sel_unt[0]
    binned_spikes[:, cc] = np.histogram(spk_times[sel_unt], bins=time_dt)[0]
    cc += 1

# %%
# select min rate
sel_units = binned_spikes.mean(axis=0) / dt_sec > 1
binned_spikes = binned_spikes[:, sel_units]
curated_units = curated_units[sel_units]

# %%
# reverse corr
binned_spikes2 = np.zeros((time_sec.shape[0] - 1, curated_units.shape[0]))

for cc, unt_id in enumerate(curated_units):
    sel_unt = np.where(unit_ids == unt_id)[0]
    assert (len(sel_unt) == 1)
    sel_unt = sel_unt[0]
    binned_spikes2[:, cc] = np.histogram(spk_times[sel_unt], bins=time_sec)[0]

# %%
average_over_frames = 3
for k in range(1, 6):
    for av in range(1, 1 + average_over_frames):
        rvs = np.zeros((51, 51))
        plt.subplot(4, 5, av)
        for idx in np.where(binned_spikes2[:, 5] != 0)[0]:
            rvs = rvs + frame_tensor[:, :, idx - av:idx].mean(axis=2) * binned_spikes2[idx, 5]
        plt.imshow(rvs)
        plt.xticks([])
        plt.yticks([])
# %%
# linear regr
ws = 5
X = np.zeros((binned_spikes2.shape[0] - ws, 51, 51, ws))
for k in range(ws, binned_spikes2.shape[0]):
    X[k - ws, :, :, :] = frame_tensor[..., k - ws: k]

# %%
# ridge
model = Ridge()
par_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
kfold = KFold(shuffle=False)
cls = GridSearchCV(model, param_grid=par_grid, cv=kfold)
# %%
# fit
neu = 5
cls.fit(X.reshape(-1, 51 ** 2 * 5), binned_spikes2[5:, neu])

# %%
#  use basis
basis = nmo.basis.RaisedCosineBasisLinear(10) ** 2
_, _, B = basis.evaluate_on_grid(51, 51)

# %%
# pass through basis (long on CPU)
Xb = np.einsum("tijk,ijm->tmk", X, B)
# %%
# fit from basis
neu = 5
cls_basis = GridSearchCV(model, param_grid=par_grid, cv=kfold)
cls_basis.fit(Xb.reshape(-1, np.prod(Xb.shape[1:])), binned_spikes2[5:, neu])
# %%
# plot results
plt.figure()
plt.title("Linear Regression")
weights = cls_basis.best_estimator_.coef_.reshape(100, 5)
imgs = np.einsum("wk,ijw->ijk", weights, B)
for k in range(5):
    plt.subplot(1, 5, k + 1)

    plt.imshow(imgs[..., k], vmin=imgs.min(), vmax=imgs.max())
    plt.xticks([])
    plt.yticks([])

# %%
# fit a glm
param_grid_nemos = {"observation_model__regularizer_strength": par_grid["alpha"]}
regularizer = nmo.regularizer.Ridge(solver_name="LBFGS",
                                    solver_kwargs={'jit': True, 'tol': 10 ** -8},
                                    )
observation_model = nmo.observation_models.PoissonObservations(inverse_link_function=jnp.exp)
model_jax = nmo.glm.GLM(regularizer=regularizer, observation_model=observation_model)
cls_basis_glm = GridSearchCV(model_jax, param_grid=par_grid, cv=kfold)
cls_basis_glm.fit(
    Xb.reshape(-1, np.prod(Xb.shape[1:]))[:, None, :],
    binned_spikes2[5:, neu:neu + 1])


# plot resutls glm
plt.figure()
plt.title("GLM")
weights = cls_basis_glm.best_estimator_.spike_basis_coeff_.reshape(100, 5)
imgs = np.einsum("wk,ijw->ijk", weights, B)
for k in range(5):
    plt.subplot(1, 5, k + 1)

    plt.imshow(imgs[..., k], vmin=imgs.min(), vmax=imgs.max())
    plt.xticks([])
    plt.yticks([])


