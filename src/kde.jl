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

end
