"""
    isj_bandwidth(data; max_order=7[, npoints])

Estimate the KDE bandwidth using the Improved Sheather-Jones (ISJ) method [^Botev2010].

ISJ is especially good for multimodal distributions with well-separated modes, but it also
works well for smooth, unimodal distributions. It uses an iterative approach to find the
bandwidth by estimating the roughness of each derivative of the KDE up to `max_order`.
This implementation is based on the reference code [^ISJRefCode], with one departure: if it
fails to find a solution to a fixed point problem, it returns the Silverman's rule of thumb
bandwidth.

Optionally `npoints` can be specified to control the number of points used in the KDE.
This can be much less than what will be used for the actual KDE.

[^Botev2010]: Kernel density estimation via diffusion.
              Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
              Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
              doi: [10.1214/10-AOS799](https://doi.org/10.1214/10-AOS799)
[^ISJRefCode]: http://web1.maths.unsw.edu.au/~zdravkobotev/php/kde_m.php
"""
function isj_bandwidth(
    data::AbstractVector{<:Real};
    npoints::Int=min(2^8, nextpow(2, length(data))),
    max_order::Int=7,
)
    n = length(data)
    data_min, data_max = extrema(data)
    data_range = data_max - data_min

    edges = range(data_min, data_max, npoints + 1)
    hist = StatsBase.fit(StatsBase.Histogram, data, edges)
    rel_counts = normalize(hist; mode=:probability).weights

    ft = @views FFTW.dct(rel_counts)[2:end]
    ft .*= sqrt(2npoints)
    t = (1:(npoints - 1)) * π

    bw_sq = _find_root(bw_sq -> _fixed_point(bw_sq, n, t, ft, max_order), n, data)

    bandwidth = sqrt(bw_sq) * data_range

    return bandwidth
end

function _fixed_point(bw_sq, n, t, ft_dens, max_order=7)
    dist = Ref(KernelDensity.kernel_dist(Distributions.Normal, sqrt(bw_sq)))
    ftj = @. t^max_order * ft_dens * Distributions.cf(dist, t)
    roughness = sum(abs2, ftj) / 2  # roughness (L2-norm) of f^{(j)}
    deriv_orders = 1:max_order
    dfact = cumprod(2 .* deriv_orders .- 1)
    for j in reverse(deriv_orders[2:(end - 1)])
        # Eq. 29
        c = (1 + 2^(-(j + 1//2))) * dfact[j]
        bw_star = (c / (3n * (sqrthalfπ * roughness)))^(1//(3 + 2j))
        dist = Ref(KernelDensity.kernel_dist(Distributions.Normal, bw_star))
        @. ftj = t^j * ft_dens * Distributions.cf(dist, t)
        roughness = sum(abs2, ftj) / 2
    end

    # Eq. 38
    return bw_sq - (2n * (sqrtπ * roughness))^(-2//5)
end

function _find_root(f::Function, n::Int, data)
    n = clamp(n, 50, 1_050)
    upper_bound = sqrt(1e-11 + 1e-5 * (n - 50))

    # if we find a bracketing interval, then find the root
    s0 = sign(f(0))
    while upper_bound < 1
        sign(f(upper_bound)) != s0 && return Roots.find_zero(f, (0, upper_bound))
        upper_bound *= sqrt2
    end

    # if we can't find a bracketing interval, then return Silverman's rule of thumb
    data_min, data_max = extrema(data)
    data_range = data_max - data_min
    return (KernelDensity.default_bandwidth(data) / data_range)^2
end

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
