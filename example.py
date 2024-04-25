# %%
import numpy as np


q = np.asarray([0, 0.5, 1])
# q *= 10
arr = np.arange(1000000).reshape(-1, 1000)
print(arr)

w = np.arange(1000)

import numpy as np


def quantile(a, q, axis=0, weights=None):
    if axis < 0:
        axis += len(a.shape)
    if weights is None:
        return np.quantile(a, q)
    weights = np.broadcast_to(weights, a.shape).astype(float)
    weights /= weights.sum(axis=axis, keepdims=True)

    ashape = a.shape
    axislen = ashape[axis]

    sorted_args = np.argsort(a, axis=axis)
    sorted_a = np.take_along_axis(a, sorted_args, axis=axis)
    sorted_w = np.take_along_axis(weights, sorted_args, axis=axis)
    # weights on the first and last

    a_swap = np.swapaxes(sorted_a, axis, -1)
    w_swap = np.swapaxes(sorted_w, axis, -1)
    a_flatten = a_swap.reshape(-1, axislen)
    w_flatten = w_swap.reshape(-1, axislen)

    # probably wanna change this into ufunc
    q_flatten = np.zeros((a_flatten.shape[0], len(q)))
    for al, wl, ql in zip(a_flatten, w_flatten, q_flatten):
        ql[:] = np.interp(q, wl.cumsum(), al)
    return q_flatten


# %%
# std = np.quantile(arr, q, axis=1, method="inverted_cdf", weights=w)
import dnp
import numpy as np


q = np.asarray([0, 0.5, 1])
# q *= 10
arr = np.arange(1000000).reshape(-1, 1000).astype(float)
arr[:, ::2] = np.nan

w = np.arange(1000)
res = dnp.nanquantile(arr, q, axis=1, method="inverted_cdf", weights=w)
print(res)
# std2 = quantile(arr, q, axis=1, weights=w)
# print(np.where(std2 != res))


# %%time


res = quantile(arr, q, axis=1, weights=w)


# %%
import warnings


def _remove_nan_1d(arr1d, overwrite_input=False):
    if arr1d.dtype == object:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = np.isnan(arr1d)
    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=6)
        return arr1d[:0], True
    elif s.size == 0:
        return arr1d, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        # select non-nans at end of array
        enonan = arr1d[-s.size :][~c[-s.size :]]
        # fill nans in beginning of array with non-nans of end
        arr1d[s[: enonan.size]] = enonan

        return arr1d[: -s.size], True


def _nanquantile_1d(a: np.array, q: np.array, w: np.array):
    # a, overwrite_input = _remove_nan_1d(a)
    if a.size == 0:
        # convert to scalar
        return np.full(q.shape, np.nan, dtype=a.dtype)[()]
    sorted_args = np.argsort(a)
    sorted_a = a[sorted_args]
    sorted_w = w[sorted_args]

    cdf = sorted_w.cumsum(dtype=np.float64)
    result = np.empty(q.shape, dtype=q.dtype)
    result[:] = np.interp(q, cdf, sorted_a)
    print(result)
    return result


def fast_nan_quantile(a, q, axis=0, weights=None):
    if axis < 0:
        axis += len(a.shape)
    if a.size == 0:
        return np.nanmean(a, axis=axis)

    if weights is None:
        return np.nanquantile(a, q)
    weights = weights.astype(float)
    weights /= weights.sum(keepdims=True)
    result = np.apply_along_axis(_nanquantile_1d, axis, a, q, weights)
    return result


fast_res = fast_nan_quantile(arr, q, axis=1, weights=w)

# %%
# print(res)
# print(fast_res)
print(np.where(res != fast_res))


# %%
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr = arr.reshape(-1, 2)
weights = np.random.rand(2)
print("init", weights)
weights = np.broadcast_to(weights, arr.shape).astype(float)
print("broadcast", weights)
weights /= weights.sum(axis=0, keepdims=True)
print("normalize", weights)
# %%
