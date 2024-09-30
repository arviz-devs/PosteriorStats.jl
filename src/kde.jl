function kde_reflected(
    data::AbstractVector{<:Real};
    bounds=extrema(data),
    npoints::Int=2_048,
    bandwidth::Real=KernelDensity.default_bandwidth(data),
    kwargs...,
)
    # set up the grid, aligning the edges with the bounds
    lo, hi, npad_lower, npad_upper = _kde_padding(
        bounds, _kde_boundary(data, bandwidth), npoints
    )
    npoints_with_padding = npoints + npad_lower + npad_upper
    midpoints = range(lo, hi; length=npoints_with_padding)

    k = KernelDensity.kde(data, midpoints; bandwidth, kwargs...)

    # reflection method
    lsplit = npad_lower + 1
    usplit = npoints_with_padding - npad_upper
    npad_lower = min(npad_lower, npoints)
    npad_upper = min(npad_upper, npoints)
    density = k.density[lsplit:usplit]
    density[1:npad_lower] .+= @view k.density[range(;
        start=lsplit - 1, step=-1, length=npad_lower
    )]
    density[(end - npad_upper + 1):end] .+= @view k.density[range(;
        stop=usplit + 1, step=-1, length=npad_upper
    )]
    x = k.x[lsplit:usplit]

    return KernelDensity.UnivariateKDE(x, density)
end

# adapt KernelDensity.kde_boundary, which is internal
function _kde_boundary(data::AbstractVector{<:Real}, bandwidth::Real)
    lo, hi = extrema(data)
    return (lo - 4 * bandwidth, hi + 4 * bandwidth)
end

function _kde_padding(data_bounds, hist_bounds, npoints)
    # set up the grid, aligning the edges with the bounds
    lower, upper = data_bounds
    lo, hi = hist_bounds
    bin_width = (upper - lower) / (npoints + 1)
    npad_lower = isfinite(lower) ? clamp(Int(cld(lower - lo, bin_width)), 0, npoints) : 0
    npad_upper = isfinite(upper) ? clamp(Int(cld(hi - upper, bin_width)), 0, npoints) : 0
    if ispow2(npoints)
        # if the user already requested a power of 2, we shouldn't sacrifice efficiency by
        # requesting a non-power of 2
        npoints_with_padding = npoints + npad_lower + npad_upper
        extra_padding = nextpow(2, npoints_with_padding) - npoints_with_padding
        a, b = divrem(extra_padding, 2)
        npad_lower += a + b
        npad_upper += a
    end
    npoints_with_padding = npoints + npad_lower + npad_upper
    lo = lower - (npad_lower - 1//2) * bin_width
    hi = upper + (npad_upper - 1//2) * bin_width
    return lo, hi, npad_lower, npad_upper
end
