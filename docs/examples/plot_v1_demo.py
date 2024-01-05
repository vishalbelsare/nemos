# %%
# Import and load
import numpy as np
import scipy.io as sio
import scipy.stats as sts
from scipy.signal import convolve2d

import matplotlib.pylab as plt
import nemos as nmo
import jax.numpy as jnp

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
from itertools import product


def find_optimal_gauss(image, pixel_size, variances, angles):
    # define point grid
    X, Y = np.meshgrid(np.arange(pixel_size), np.arange(pixel_size))
    xy = np.c_[X.flatten(), Y.flatten()]
    # define rotation
    rotation = lambda angle: np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]]
    )
    # define gaussian
    gauss = lambda x, cov: sts.multivariate_normal.pdf(
        x,
        mean=[pixel_size // 2, pixel_size // 2],
        cov=(cov + cov.T)/2).reshape(
        pixel_size, pixel_size
    )
    rot_gauss = lambda x, ang, var_x, var_y: gauss(x, rotation(ang).T @ np.diag([var_x, var_y]) @ rotation(ang))
    max_val = 0
    best_param = None
    best_coord = None
    best_kernel = None
    all_res = []
    for var_x, var_y, ang in product(variances, variances, angles):
        kernel = rot_gauss(xy, ang, var_x, var_y)
        kernel = kernel / np.linalg.norm(kernel)
        convolved_image = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
        peak_x, peak_y = np.unravel_index(np.argmax(convolved_image), image.shape)
        all_res.append((var_x, var_y, ang, np.max(convolved_image), convolved_image[peak_x, peak_y]))
        if max_val < convolved_image[peak_x, peak_y]:
            best_param = var_x, var_y, ang
            best_coord = peak_x, peak_y
            best_kernel = kernel
            max_val = convolved_image[peak_x, peak_y]
    return best_param, best_kernel, best_coord,all_res


def plot_image_with_contour(image, kernel, coord, pad_with_nan=True):
    """
    Plots an image with a contour overlay of the kernel.

    Parameters:
    image : np.ndarray
        The image to be plotted.
    kernel : np.ndarray
        The kernel whose contours are to be overlaid on the image.
    coord : tuple
        The coordinates (x, y) where the kernel is centered.
    pad_with_nan : bool
        If True, pads with NaN, otherwise pads with zeros.
    """
    plt.figure(figsize=(8, 8))

    # Pad the kernel to match the image size
    pad_height = image.shape[0] - kernel.shape[0]
    pad_width = image.shape[1] - kernel.shape[1]
    pad_top = coord[0] - kernel.shape[0] // 2
    pad_left = coord[1] - kernel.shape[1] // 2
    pad_bottom = pad_height - pad_top
    pad_right = pad_width - pad_left

    if pad_with_nan:
        pad_value = np.nan
    else:
        pad_value = 0

    padded_kernel = np.pad(
        kernel, ((pad_top, pad_bottom), (pad_left, pad_right)),
        'constant', constant_values=(pad_value, pad_value))

    # Plot the image
    plt.imshow(image, cmap='gray')  # Choose a colormap suitable for your image

    # Overlay the contour. Adjust levels and colors as needed.
    plt.contour(padded_kernel, levels=5, colors='red', alpha=0.6)  # alpha for transparency

    # Customizations
    plt.title("Image with Kernel Contour")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.show()


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
# reverse correlation
plt.figure()
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
# Fit
neu = 5
regularizer_strength = 0.01
# %%
#  use basis
basis = nmo.basis.RaisedCosineBasisLinear(10) ** 2
_, _, B = basis.evaluate_on_grid(51, 51)

# %%
# pass through basis (long on CPU)
Xb = jnp.einsum("tijk,ijm->tmk", X, B)
# %%
# fit from basis
model = Ridge(alpha=regularizer_strength)
model.fit(Xb.reshape(-1, np.prod(Xb.shape[1:])), binned_spikes2[5:, neu])
# %%
# plot results
plt.figure()
plt.title("Linear Regression")
weights = model.coef_.reshape(100, 5)
imgs = np.einsum("wk,ijw->ijk", weights, B)
for k in range(5):
    plt.subplot(1, 5, k + 1)

    plt.imshow(imgs[..., k], vmin=imgs.min(), vmax=imgs.max())
    plt.xticks([])
    plt.yticks([])

# %%
# fit a glm
regularizer = nmo.regularizer.Ridge(solver_name="LBFGS",
                                    solver_kwargs={'jit': True, 'tol': 10 ** -8},
                                    regularizer_strength=regularizer_strength)
observation_model = nmo.observation_models.PoissonObservations(inverse_link_function=jnp.exp)
model_jax = nmo.glm.GLM(regularizer=regularizer, observation_model=observation_model)
model_jax.fit(
    Xb.reshape(-1, np.prod(Xb.shape[1:]))[:, None, :],
    binned_spikes2[5:, neu:neu + 1])


# plot resutls glm
weights = model_jax.coef_.reshape(100, 5)
plt.figure()
plt.title("GLM")
imgs = np.einsum("wk,ijw->ijk", weights, B)
for k in range(5):
    plt.subplot(1, 5, k + 1)

    plt.imshow(imgs[..., k], vmin=imgs.min(), vmax=imgs.max())
    plt.xticks([])
    plt.yticks([])


# create filter bank
rot_angles = np.arange(0, 20) * np.pi / 20
vars = np.linspace(0.5, 10.5, 20)
pixel_size = 17
best_param, best_kernel, best_coord, all_res = find_optimal_gauss(-imgs[..., 3], pixel_size, vars, rot_angles)
plot_image_with_contour(imgs[..., 3], best_kernel, best_coord)