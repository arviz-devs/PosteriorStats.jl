const STD_NORM_IQR = rationalize(1.34)

"""
    bandwidth_silverman(x; kwargs...) -> Real
"""
function bandwidth_silverman(
    x::AbstractVector{<:Real}; alpha::Real=9//10, std::Real=Statistics.std(x)
)
    n = length(x)
    iqr = StatsBase.iqr(x)
    quantile_width = iqr / STD_NORM_IQR
    width = min(std, quantile_width)
    T = typeof(one(width))
    return alpha * width * T(n)^(-1//5)
end

function _padding_factor(
    kernel::Distributions.ContinuousUnivariateDistribution, prob_tail::Real
)
    return Int(cld(Distributions.cquantile(kernel, prob_tail / 2), Statistics.std(kernel)))
end

function _kernel_with_bandwidth(
    T::Type{<:Distributions.ContinuousUnivariateDistribution}, bw
)
    d = T()
    dcentered = (d - Statistics.median(d)) * bw / Statistics.std(d)
    StatsBase.skewness(dcentered) ≈ 0 || throw(ArgumentError("Kernel must be symmetric."))
    return dcentered
end

"""
    kde(x; kwargs...) -> KernelDensity.UnivariateKDE

Compute the univariate kernel density estimate of data `x`.

# Arguments
- `x`: data array

# Keyword arguments
- `bandwidth::Real`: bandwidth of the kernel. Defaults to [`bandwidth_silverman(x)`](@ref).
- `kernel::Type{<:Distributions.ContinuousUnivariateDistribution}`: type of kernel to build.
    Defaults to `Normal`.
- `bound_correction::Bool`: whether to perform boundary correction. Defaults to `true`.
    If `false`, the resulting truncated KDE is not normalized to 1.
- `npoints::Int`: number of points at which the resulting KDE is evaluated. Defaults to
    `512`.
"""
function kde(
    x::AbstractVector;
    bandwidth::Real=bandwidth_silverman(x),
    kernel=Distributions.Normal,
    bound_correction::Bool=true,
    npoints::Int=512,
    pad_factor::Union{Real,Nothing}=nothing,
)
    grid_size = max(npoints, 100)
    grid_min, grid_max = extrema(x)
    bin_width = (grid_max - grid_min) / grid_size

    if pad_factor === nothing
        # work out how much padding to add to guarantee that extra density due to wraparound
        # is negligible
        prob_tail = 1e-3
        _kernel = _kernel_with_bandwidth(kernel, bandwidth)
        _pad_factor = _padding_factor(_kernel, prob_tail)
    elseif pad_factor < 0
        throw(DomainError(pad_factor, "Padding factor must be non-negative."))
    else
        _pad_factor = pad_factor
    end
    # always pad by at least 1 bin on each side to ensure that the boundary passed to kde
    # contains all data points (otherwise, they will be ignored)
    grid_pad_size = 2 * max(1, Int(cld(_pad_factor * bandwidth, bin_width)))
    npad_left = npad_right = grid_pad_size ÷ 2

    # pad to avoid wraparound at the boundary
    grid_min -= bin_width * npad_left
    grid_max += bin_width * npad_right
    grid_size += grid_pad_size

    # compute density
    boundary = (grid_min + bin_width / 2, grid_max - bin_width / 2)
    k = KernelDensity.kde(x; npoints=grid_size, boundary, bandwidth, kernel)
    midpoints = k.x
    density = k.density

    if bound_correction
        nbin_reflect = min(npad_left, npad_right)
        il = firstindex(density) + npad_left
        ir = lastindex(density) - npad_right
        # reflect density at the boundary (x_min, x_max)
        density[range(il; length=nbin_reflect)] .+= @view density[range(
            il - 1; step=-1, length=nbin_reflect
        )]
        density[range(ir; step=-1, length=nbin_reflect)] .+= @view density[range(
            ir + 1; length=nbin_reflect
        )]
    end

    # remove padding
    midpoints = midpoints[(begin + npad_left):(end - npad_right)]
    density = density[(begin + npad_left):(end - npad_right)]

    return KernelDensity.UnivariateKDE(midpoints, density)
end

struct UnivariateCKDE{X,P}
    x::X
    probability::P
end

"""
    ckde(x; kwargs...) -> UnivariateCKDE

Compute the CDF of the kernel density estimate of data `x`.

For details about arguments and keywords, see [`kde`](@ref)
"""
ckde(x; kwargs...) = ckde(kde(x; kwargs...))

"""
    ckde(k::KernelDensity.UnivariateKDE) -> UnivariateCKDE

Compute the CDF of the provided kernel density estimate.
"""
function ckde(k::KernelDensity.UnivariateKDE)
    prob = cumsum(k.density) .* step(k.x)
    return UnivariateCKDE(k.x, prob ./ prob[end])
end
