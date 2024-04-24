import functools
import warnings
import numpy as np
import operator


# array_function_dispatch = functools.partial(
#     overrides.array_function_dispatch, module="numpy"
# )
def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    """
    Compute the floating point indexes of an array for the linear
    interpolation of quantiles.
    n : array_like
        The sample sizes.
    quantiles : array_like
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    alpha and beta values depend on the chosen method
    (see quantile documentation)

    Reference:
    Hyndman&Fan paper "Sample Quantiles in Statistical Packages",
    DOI: 10.1080/00031305.1996.10473566
    """
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1


def _get_gamma(virtual_indexes, previous_indexes, method):
    """
    Compute gamma (a.k.a 'm' or 'weight') for the linear interpolation
    of quantiles.

    virtual_indexes : array_like
        The indexes where the percentile is supposed to be found in the sorted
        sample.
    previous_indexes : array_like
        The floor values of virtual_indexes.
    interpolation : dict
        The interpolation method chosen, which may have a specific rule
        modifying gamma.

    gamma is usually the fractional part of virtual_indexes but can be modified
    by the interpolation method.
    """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method["fix_gamma"](gamma, virtual_indexes)
    # Ensure both that we have an array, and that we keep the dtype
    # (which may have been matched to the input array).
    return np.asanyarray(gamma, dtype=virtual_indexes.dtype)


def _lerp(a, b, t, out=None):
    """
    Compute the linear interpolation weighted by gamma on each point of
    two same shape array.

    a : array_like
        Left bound.
    b : array_like
        Right bound.
    t : array_like
        The interpolation weight.
    out : array_like
        Output array.
    """
    diff_b_a = np.subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    lerp_interpolation = np.asanyarray(np.add(a, diff_b_a * t, out=out))
    np.subtract(
        b,
        diff_b_a * (1 - t),
        out=lerp_interpolation,
        where=t >= 0.5,
        casting="unsafe",
        dtype=type(lerp_interpolation.dtype),
    )
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation


def _get_gamma_mask(shape, default_value, conditioned_value, where):
    out = np.full(shape, default_value)
    np.copyto(out, conditioned_value, where=where, casting="unsafe")
    return out


def _discret_interpolation_to_boundaries(index, gamma_condition_fun):
    previous = np.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(
        shape=index.shape,
        default_value=next,
        conditioned_value=previous,
        where=gamma_condition_fun(gamma, index),
    ).astype(np.intp)
    # Some methods can lead to out-of-bound integers, clip them:
    res[res < 0] = 0
    return res


def _closest_observation(n, quantiles):
    gamma_fun = lambda gamma, index: (gamma == 0) & (np.floor(index) % 2 == 0)
    return _discret_interpolation_to_boundaries((n * quantiles) - 1 - 0.5, gamma_fun)


def _inverted_cdf(n, quantiles):
    gamma_fun = lambda gamma, _: (gamma == 0)
    return _discret_interpolation_to_boundaries((n * quantiles) - 1, gamma_fun)


def normalize_axis_index(axis: int, ndim: int, msg_prefix: str):
    if axis < -ndim or axis >= ndim:
        raise np.AxisError(
            "{}: axis {} is out of bounds for array of dimension {}".format(
                msg_prefix, axis, ndim
            )
        )
    if axis < 0:
        axis += ndim
    return axis


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.

    Used internally by multi-axis-checking logic.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated

    See also
    --------
    normalize_axis_index : normalizing a single scalar axis
    """
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError("repeated axis in `{}` argument".format(argname))
        else:
            raise ValueError("repeated axis")
    return axis


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (q.min() >= 0 and q.max() <= 1):
            return False
    return True


def _weights_are_valid(weights, a, axis):
    """Validate weights array.

    We assume, weights is not None.
    """
    wgt = np.asanyarray(weights)

    # Sanity checks
    if a.shape != wgt.shape:
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of a and weights " "differ."
            )
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            raise ValueError(
                "Shape of weights must be consistent with "
                "shape of a along specified axis."
            )

        # setup wgt to broadcast along axis
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(
            tuple((s if ax in axis else 1) for ax, s in enumerate(a.shape))
        )
    return wgt


# def _nanquantile_dispatcher(
#     a,
#     q,
#     axis=None,
#     out=None,
#     overwrite_input=None,
#     method=None,
#     keepdims=None,
#     *,
#     weights=None,
#     interpolation=None,
# ):
#     return (a, q, out, weights)


def _remove_nan_1d(arr1d, weights, overwrite_input=False):
    """
    Equivalent to arr1d[~arr1d.isnan()], but in a different order

    Presumably faster as it incurs fewer copies

    Parameters
    ----------
    arr1d : ndarray
        Array to remove nans from
    overwrite_input : bool
        True if `arr1d` can be modified in place

    Returns
    -------
    res : ndarray
        Array with nan elements removed
    overwrite_input : bool
        True if `res` can be modified in place, given the constraint on the
        input
    """
    if arr1d.dtype == object:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = np.isnan(arr1d)

    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=6)
        if weights is None:
            return arr1d[:0], None, True
        else:
            return arr1d[:0], weights[:0], True
    elif s.size == 0:
        return arr1d, weights, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        # select non-nans at end of array
        enonan = arr1d[-s.size :][~c[-s.size :]]
        # fill nans in beginning of array with non-nans of end
        arr1d[s[: enonan.size]] = enonan

        if weights is None:
            return arr1d[: -s.size], None, True
        else:
            if not overwrite_input:
                weights = weights.copy()
            print(arr1d.shape, weights.shape)
            enonan = weights[-s.size :][~c[-s.size :]]
            weights[s[: enonan.size]] = enonan

            return arr1d[: -s.size], weights[: -s.size], True


_QuantileMethods = dict(
    # --- HYNDMAN and FAN METHODS
    # Discrete methods
    inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: _inverted_cdf(n, quantiles),
        fix_gamma=None,  # should never be called
    ),
    averaged_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: (n * quantiles) - 1,
        fix_gamma=lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape,
            default_value=1.0,
            conditioned_value=0.5,
            where=gamma == 0,
        ),
    ),
    closest_observation=dict(
        get_virtual_index=lambda n, quantiles: _closest_observation(n, quantiles),
        fix_gamma=None,  # should never be called
    ),
    # Continuous methods
    interpolated_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 0, 1
        ),
        fix_gamma=lambda gamma, _: gamma,
    ),
    hazen=dict(
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 0.5, 0.5
        ),
        fix_gamma=lambda gamma, _: gamma,
    ),
    weibull=dict(
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 0, 0
        ),
        fix_gamma=lambda gamma, _: gamma,
    ),
    # Default method.
    # To avoid some rounding issues, `(n-1) * quantiles` is preferred to
    # `_compute_virtual_index(n, quantiles, 1, 1)`.
    # They are mathematically equivalent.
    linear=dict(
        get_virtual_index=lambda n, quantiles: (n - 1) * quantiles,
        fix_gamma=lambda gamma, _: gamma,
    ),
    median_unbiased=dict(
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 1 / 3.0, 1 / 3.0
        ),
        fix_gamma=lambda gamma, _: gamma,
    ),
    normal_unbiased=dict(
        get_virtual_index=lambda n, quantiles: _compute_virtual_index(
            n, quantiles, 3 / 8.0, 3 / 8.0
        ),
        fix_gamma=lambda gamma, _: gamma,
    ),
    # --- OTHER METHODS
    lower=dict(
        get_virtual_index=lambda n, quantiles: np.floor((n - 1) * quantiles).astype(
            np.intp
        ),
        fix_gamma=None,  # should never be called, index dtype is int
    ),
    higher=dict(
        get_virtual_index=lambda n, quantiles: np.ceil((n - 1) * quantiles).astype(
            np.intp
        ),
        fix_gamma=None,  # should never be called, index dtype is int
    ),
    midpoint=dict(
        get_virtual_index=lambda n, quantiles: 0.5
        * (np.floor((n - 1) * quantiles) + np.ceil((n - 1) * quantiles)),
        fix_gamma=lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape,
            default_value=0.5,
            conditioned_value=0.0,
            where=index % 1 == 0,
        ),
    ),
    nearest=dict(
        get_virtual_index=lambda n, quantiles: np.around((n - 1) * quantiles).astype(
            np.intp
        ),
        fix_gamma=None,
        # should never be called, index dtype is int
    ),
)


def _get_indexes(arr, virtual_indexes, valid_values_count):
    """
    Get the valid indexes of arr neighbouring virtual_indexes.
    Note
    This is a companion function to linear interpolation of
    Quantiles

    Returns
    -------
    (previous_indexes, next_indexes): Tuple
        A Tuple of virtual_indexes neighbouring indexes
    """
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    next_indexes = np.asanyarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


def _quantile(
    arr: np.array,
    quantiles: np.array,
    axis: int = -1,
    method="linear",
    out=None,
    weights=None,
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    It computes the quantiles of the array for the given axis.
    A linear interpolation is performed based on the `interpolation`.

    By default, the method is "linear" where alpha == beta == 1 which
    performs the 7th method of Hyndman&Fan.
    With "median_unbiased" we get alpha == beta == 1/3
    thus the 8th method of Hyndman&Fan.
    """
    # --- Setup
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    if axis != 0:  # But moveaxis is slow, so only call it if necessary.
        arr = np.moveaxis(arr, axis, destination=0)
    supports_nans = np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in "Mm"

    if weights is None:
        # --- Computation of indexes
        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not an valid index.
        # The nearest neighbours are used for interpolation
        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(
                f"{method!r} is not a valid method. Use one of: "
                f"{_QuantileMethods.keys()}"
            ) from None
        virtual_indexes = method_props["get_virtual_index"](values_count, quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype, np.integer)
            supports_integers = method == "linear" and int_virtual_indices

        if supports_integers:
            # No interpolation needed, take the points along axis
            if supports_nans:
                # may contain nan, which would sort to the end
                arr.partition(
                    np.concatenate((virtual_indexes.ravel(), [-1])),
                    axis=0,
                )
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = np.take(arr, virtual_indexes, axis=0, out=out)
        else:
            previous_indexes, next_indexes = _get_indexes(
                arr, virtual_indexes, values_count
            )
            # --- Sorting
            arr.partition(
                np.unique(
                    np.concatenate(
                        (
                            [0, -1],
                            previous_indexes.ravel(),
                            next_indexes.ravel(),
                        )
                    )
                ),
                axis=0,
            )
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None
            # --- Get values from indexes
            previous = arr[previous_indexes]
            next = arr[next_indexes]
            # --- Linear interpolation
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props)
            result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
            gamma = gamma.reshape(result_shape)
            result = _lerp(previous, next, gamma, out=out)
    else:
        # Weighted case
        # This implements method="inverted_cdf", the only supported weighted
        # method, which needs to sort anyway.
        weights = np.asanyarray(weights)
        if axis != 0:
            weights = np.moveaxis(weights, axis, destination=0)
        index_array = np.argsort(arr, axis=0, kind="stable")

        # arr = arr[index_array, ...]  # but this adds trailing dimensions of
        # 1.
        arr = np.take_along_axis(arr, index_array, axis=0)
        if weights.shape == arr.shape:
            weights = np.take_along_axis(weights, index_array, axis=0)
        else:
            # weights is 1d
            weights = weights.reshape(-1)[index_array, ...]

        if supports_nans:
            # may contain nan, which would sort to the end
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            # cannot contain nan
            slices_having_nans = np.array(False, dtype=bool)

        # We use the weights to calculate the empirical cumulative
        # distribution function cdf
        cdf = weights.cumsum(axis=0, dtype=np.float64)
        cdf /= cdf[-1, ...]  # normalization to 1
        # Search index i such that
        #   sum(weights[j], j=0..i-1) < quantile <= sum(weights[j], j=0..i)
        # is then equivalent to
        #   cdf[i-1] < quantile <= cdf[i]
        # Unfortunately, searchsorted only accepts 1-d arrays as first
        # argument, so we will need to iterate over dimensions.

        # Without the following cast, searchsorted can return surprising
        # results, e.g.
        #   np.searchsorted(np.array([0.2, 0.4, 0.6, 0.8, 1.]),
        #                   np.array(0.4, dtype=np.float32), side="left")
        # returns 2 instead of 1 because 0.4 is not binary representable.
        if quantiles.dtype.kind == "f":
            cdf = cdf.astype(quantiles.dtype)

        def find_cdf_1d(arr, cdf):
            indices = np.searchsorted(cdf, quantiles, side="left")
            # We might have reached the maximum with i = len(arr), e.g. for
            # quantiles = 1, and need to cut it to len(arr) - 1.
            indices = np.minimum(indices, values_count - 1)
            result = np.take(arr, indices, axis=0)
            return result

        r_shape = arr.shape[1:]
        if quantiles.ndim > 0:
            r_shape = quantiles.shape + r_shape
        if out is None:
            result = np.empty_like(arr, shape=r_shape)
        else:
            if out.shape != r_shape:
                msg = (
                    f"Wrong shape of argument 'out', shape={r_shape} is "
                    f"required; got shape={out.shape}."
                )
                raise ValueError(msg)
            result = out

        # See apply_along_axis, which we do for axis=0. Note that Ni = (,)
        # always, so we remove it here.
        Nk = arr.shape[1:]
        for kk in np.ndindex(Nk):
            result[(...,) + kk] = find_cdf_1d(arr[np.s_[:,] + kk], cdf[np.s_[:,] + kk])

        # Make result the same as in unweighted inverted_cdf.
        if result.shape == () and result.dtype == np.dtype("O"):
            result = result.item()

    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar, but indexing will be correct
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result


def _quantile_ureduce_func(
    a: np.array,
    q: np.array,
    weights: np.array,
    axis: int = None,
    out=None,
    overwrite_input: bool = False,
    method="linear",
) -> np.array:
    if q.ndim > 1:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    else:
        if axis is None:
            axis = 0
            arr = a.flatten()
            wgt = None if weights is None else weights.flatten()
        else:
            arr = a.copy()
            wgt = weights
    result = _quantile(arr, quantiles=q, axis=axis, method=method, out=out, weights=wgt)
    return result


def _quantile_unchecked(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    weights=None,
):
    """Assumes that q is in [0, 1], and is an ndarray"""
    return _ureduce(
        a,
        func=_quantile_ureduce_func,
        q=q,
        weights=weights,
        keepdims=keepdims,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
    )


def _ureduce(a, func, keepdims=False, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.

    Returns result and a.shape with axis dims set to 1.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.

    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.

    """
    a = np.asanyarray(a)
    axis = kwargs.get("axis", None)
    out = kwargs.get("out", None)

    if keepdims is np._NoValue:
        keepdims = False

    nd = a.ndim
    if axis is not None:
        axis = normalize_axis_tuple(axis, nd)

        if keepdims:
            if out is not None:
                index_out = tuple(0 if i in axis else slice(None) for i in range(nd))
                kwargs["out"] = out[(Ellipsis,) + index_out]

        if len(axis) == 1:
            kwargs["axis"] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs["axis"] = -1
    else:
        if keepdims:
            if out is not None:
                index_out = (0,) * nd
                kwargs["out"] = out[(Ellipsis,) + index_out]

    r = func(a, **kwargs)

    if out is not None:
        return out

    if keepdims:
        if axis is None:
            index_r = (np.newaxis,) * nd
        else:
            index_r = tuple(np.newaxis if i in axis else slice(None) for i in range(nd))
        r = r[(Ellipsis,) + index_r]

    return r


def _nanquantile_1d(
    arr1d,
    q,
    overwrite_input=False,
    method="linear",
    weights=None,
):
    """
    Private function for rank 1 arrays. Compute quantile ignoring NaNs.
    See nanpercentile for parameter usage
    """

    arr1d, weights, overwrite_input = _remove_nan_1d(
        arr1d, weights, overwrite_input=overwrite_input
    )

    if arr1d.size == 0:
        # convert to scalar
        return np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]

    return _quantile_unchecked(
        arr1d,
        q,
        overwrite_input=overwrite_input,
        method=method,
        weights=weights,
    )


def _nanquantile_ureduce_func(
    a: np.array,
    q: np.array,
    weights: np.array,
    axis: int = None,
    out=None,
    overwrite_input: bool = False,
    method="linear",
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        wgt = None if weights is None else weights.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, method, weights=wgt)
    else:
        result = np.apply_along_axis(
            _nanquantile_1d, axis, a, q, overwrite_input, method, weights
        )
        # apply_along_axis fills in collapsed axis with results.
        # Move that axis to the beginning to match percentile's
        # convention.
        if q.ndim != 0:
            result = np.moveaxis(result, axis, 0)

    if out is not None:
        out[...] = result
    return result


def _nanquantile_ureduce_func(
    a: np.array,
    q: np.array,
    weights: np.array,
    axis: int = None,
    out=None,
    overwrite_input: bool = False,
    method="linear",
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        wgt = None if weights is None else weights.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, method, weights=wgt)
    else:
        result = np.apply_along_axis(
            _nanquantile_1d, axis, a, q, overwrite_input, method, weights
        )
        # apply_along_axis fills in collapsed axis with results.
        # Move that axis to the beginning to match percentile's
        # convention.
        if q.ndim != 0:
            result = np.moveaxis(result, axis, 0)

    if out is not None:
        out[...] = result
    return result


def _nanquantile_unchecked(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=np._NoValue,
    weights=None,
):
    """Assumes that q is in [0, 1], and is an ndarray"""
    # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    return _ureduce(
        a,
        func=_nanquantile_ureduce_func,
        q=q,
        weights=weights,
        keepdims=keepdims,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
    )


# @array_function_dispatch(_nanquantile_dispatcher)
def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=np._NoValue,
    *,
    weights=None,
):
    """
    Compute the qth quantile of the data along the specified axis,
    while ignoring nan values.
    Returns the qth quantile(s) of the array elements.

    .. versionadded:: 1.15.0

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array, containing
        nan values to be ignored
    q : array_like of float
        Probability or sequence of probabilities for the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

        If this is anything but the default value it will be passed
        through (in the special case of an empty array) to the
        `mean` function of the underlying array.  If the array is
        a sub-class and `mean` does not have the kwarg `keepdims` this
        will raise a RuntimeError.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the quantile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single probability and `axis=None`, then the result
        is a scalar. If multiple probability levels are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    quantile
    nanmean, nanmedian
    nanmedian : equivalent to ``nanquantile(..., 0.5)``
    nanpercentile : same as nanquantile, but with q in the range [0, 100].

    Notes
    -----
    For more information please see `numpy.quantile`

    Examples
    --------
    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
    >>> a[0][1] = np.nan
    >>> a
    array([[10.,  nan,   4.],
          [ 3.,   2.,   1.]])
    >>> np.quantile(a, 0.5)
    np.float64(nan)
    >>> np.nanquantile(a, 0.5)
    3.0
    >>> np.nanquantile(a, 0.5, axis=0)
    array([6.5, 2. , 2.5])
    >>> np.nanquantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.nanquantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.nanquantile(a, 0.5, axis=0, out=out)
    array([6.5, 2. , 2.5])
    >>> m
    array([6.5,  2. ,  2.5])
    >>> b = a.copy()
    >>> np.nanquantile(b, 0.5, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float).
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if weights is not None:
        # if method != "inverted_cdf":
        #     msg = "Only method 'inverted_cdf' supports weights. " f"Got: {method}."
        #     raise ValueError(msg)
        if axis is not None:
            axis = normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
    print(weights.shape)
    return _nanquantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights
    )
