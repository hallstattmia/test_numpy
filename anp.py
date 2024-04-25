import numpy as np


def quantile(a, q, axis=0, weights=None):
    if axis < 0:
        axis += len(a.shape)
    if weights is None:
        return np.quantile(a, q)
    weights = weights.transpose(np.argsort(axis))
    weights = weights.reshape(
        tuple((s if ax in np.asanyarray(axis) else 1) for ax, s in enumerate(a.shape))
    )
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

    ql_reshape = q_flatten.swapaxes(0, 1)
    ql_reshape = ql_reshape.reshape((len(q),) + ashape[:axis] + ashape[axis + 1 :])
    return ql_reshape
