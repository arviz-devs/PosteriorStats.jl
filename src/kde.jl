"""
    kde_reflected(data::AbstractVector{<:Real}; bounds=extrema(data), kwargs...)

Compute the boundary-corrected kernel density estimate (KDE) of `data` using reflection.

For ``x \\in (l, u)``, the reflected KDE has the density
```math
\\hat{f}_R(x) = \\hat{f}(x) + \\hat{f}(2l - x) + \\hat{f}(2u - x),
```
where ``\\hat{f}`` is the usual KDE of `data`. This is equivalent to augmeenting the
original data with 2 additional copies of the data reflected around each bound, computing
the usual KDE, trimming the KDE to the bounds, and renormalizing.

Any non-finite `bounds` are ignored. Remaining `kwargs` are passed to `KernelDensity.kde`.
"""
function kde_reflected(
    data::AbstractVector{<:Real};
    bounds::Union{Nothing,Tuple{Real,Real}}=nothing,
    npoints::Int=2_048,
    bandwidth::Real=KernelDensity.default_bandwidth(data),
    kwargs...,
)
    _bounds = _get_check_bounds(bounds, extrema(data))
    midpoints, idx_bulk = _kde_padded_grid(_bounds, _kde_boundary(data, bandwidth), npoints)
    kde = KernelDensity.kde(data, midpoints; bandwidth, kwargs...)
    kde_reflect = _kde_reflection(kde, idx_bulk)
    return kde_reflect
end

function _get_check_bounds(bounds, data_bounds)
    lb, ub = bounds
    lb = isfinite(lb) ? lb : oftype(lb, -Inf)
    ub = isfinite(ub) ? ub : oftype(ub, Inf)
    lb < ub || throw(
        ArgumentError(
            "Invalid bounds: $bounds. The lower bound must be less than the upper bound.",
        ),
    )
    xmin, xmax = data_bounds
    lb ≤ xmin || throw(DomainError(lb, "Some data points are below the lower bound."))
    xmax ≤ ub || throw(DomainError(ub, "Some data points are above the upper bound."))
    return (lb, ub)
end
_get_check_bounds(::Nothing, data_bounds) = data_bounds

# adapt KernelDensity.kde_boundary, which is internal
function _kde_boundary(data::AbstractVector{<:Real}, bandwidth::Real)
    lo, hi = extrema(data)
    return (lo - 4 * bandwidth, hi + 4 * bandwidth)
end

function _kde_padded_grid(bounds, hist_bounds, npoints)
    # set up the grid, aligning the edges with the bounds
    lb, ub = bounds
    lh, uh = hist_bounds
    (left, left_is_bound) = isfinite(lb) ? (lb, true) : (lh, false)
    (right, right_is_bound) = isfinite(ub) ? (ub, true) : (uh, false)
    bin_width = (right - left) / (npoints - 1 + (left_is_bound + right_is_bound) / 2)
    npad_left = clamp(
        Int(cld(left - lh + (1 - left_is_bound//2) * bin_width, bin_width)), 0, npoints
    )
    npad_right = clamp(
        Int(cld(uh - right + (1 - right_is_bound//2) * bin_width, bin_width)), 0, npoints
    )
    if ispow2(npoints)
        # if the user already requested a power of 2, we shouldn't sacrifice efficiency by
        # requesting a non-power of 2
        npoints_with_padding = npoints + npad_left + npad_right
        extra_padding = nextpow(2, npoints_with_padding) - npoints_with_padding
        npad_common, npad_rem = divrem(extra_padding, 2)
        npad_left += npad_common + npad_rem
        npad_right += npad_common
    end
    npoints_with_padding = npoints + npad_left + npad_right
    lh = left - (npad_left - left_is_bound//2) * bin_width
    uh = right + (npad_right - right_is_bound//2) * bin_width

    midpoints = range(lh, uh; length=npoints_with_padding)

    idx_left = npad_left + 1
    idx_right = npad_left + npoints
    idx_bulk = idx_left:idx_right

    return midpoints, idx_bulk
end

function _kde_reflection(kde, idx_bulk)
    npoints_with_padding = length(kde.x)
    npoints = length(idx_bulk)
    x = kde.x[idx_bulk]

    # avoid double reflections by trimming the padding to the number of points
    npad_lower = min(npoints, first(idx_bulk) - 1)
    npad_upper = min(npoints, npoints_with_padding - last(idx_bulk))

    # get idx ranges for the density that will be reflected
    idx_lower = range(; start=first(idx_bulk) - 1, step=-1, length=npad_lower)
    idx_upper = range(; stop=last(idx_bulk) + 1, step=-1, length=npad_upper)

    # reflect the density
    density = kde.density[idx_bulk]
    density[1:npad_lower] .+= view(kde.density, idx_lower)
    density[(end - npad_upper + 1):end] .+= view(kde.density, idx_upper)

    return KernelDensity.UnivariateKDE(x, density)
end
