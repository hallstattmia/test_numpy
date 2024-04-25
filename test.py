# std = np.quantile(arr, q, axis=1, method="inverted_cdf", weights=w)
import dnp
import anp
import numpy as np


q = np.asarray([0.3, 0.7])
# q *= 10
N = 10000
M = 100

arr = np.arange(N).reshape(-1, M).astype(float)
arr[:, ::2] = np.nan
arr1 = arr[:, 1::2]
w = np.arange(M)
w1 = w[1::2]

# test with np.nanquantile
std = np.quantile(arr1, q, axis=1, method="inverted_cdf", weights=w1)
res = dnp.nanquantile(arr, q, axis=1, method="inverted_cdf", weights=w)
print(np.where(res != std))


# test with anp.quantile

N = 10000
M = 100

arr = np.arange(N).reshape(-1, M).astype(float)
arr[:, ::2] = np.nan
arr1 = arr[:, 1::2]
w = np.arange(M)
w1 = w[1::2]
std = anp.quantile(arr1, q, axis=1, weights=w1)
res = dnp.nanquantile(arr, q, axis=1, method="linear", weights=w)
print(np.where(np.isclose(res, std) == False))

# test with anp.quantile
N = 10000
M = 100

arr = np.arange(N).reshape(-1, M).astype(float)
arr[::2, :] = np.nan
arr1 = arr[1::2, :]
w = np.arange(N // M)
w1 = w[1::2]

std = anp.quantile(arr1, q, axis=0, weights=w1)
res = dnp.nanquantile(arr, q, axis=0, method="linear", weights=w)
print(np.where(np.isclose(res, std) == False))

# test with perf
N = 40000000
M = 10000

arr = np.arange(N).reshape(-1, M).astype(float)
w = np.arange(M)

import time

st = time.time()
std = np.nanquantile(arr, q, axis=1, method="inverted_cdf", weights=w)
print("std time", time.time() - st)
st = time.time()
res = dnp.nanquantile(arr, q, axis=1, method="linear", weights=w)
print("res time", time.time() - st)


# test numpy1
import numpy as np

q = np.asarray([0.3, 0.7])
N = 10000000
M = 10000
arr = np.arange(N).reshape(-1, M).astype(float)
w = np.ones(M)

std = np.nanquantile(arr, q, axis=1)

# test numpy2 edge case w[0] > 0

import numpy as np

q = np.asarray([0.3, 0.7])
N = 10
M = 10
arr = np.arange(N).reshape(-1, M).astype(float)
w = np.ones(M)

std = np.nanquantile(arr, q, axis=1, method="inverted_cdf", weights=w)
