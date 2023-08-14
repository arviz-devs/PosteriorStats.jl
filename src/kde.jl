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
- `grid_length::Int`: number of grid points to use. Defaults to `512`.
"""
function kde(
    x::AbstractVector;
    bandwidth::Real=bandwidth_silverman(x),
    kernel=Distributions.Normal,
    bound_correction::Bool=true,
    grid_length::Int=512,
)
    x_min, x_max = extrema(x)

    grid_length = max(grid_length, 100)
    grid_min = x_min
    grid_max = x_max
    bin_width = (grid_max - grid_min) / grid_length

    # work out how much padding to add to guarantee that extra density due to wraparound
    # is negligible
    prob_tail = 1e-3
    _kernel = _kernel_with_bandwidth(kernel, bandwidth)
    npad_bw = _padding_factor(_kernel, prob_tail)
    npad_bin = Int(cld(npad_bw * bandwidth, bin_width))

    # add extra padding if performing boundary correction
    if bound_correction
        nbin_reflect = npad_bin
        npad_bin += nbin_reflect
    end

    # pad to avoid wraparound at the boundary
    grid_min -= bin_width * npad_bin
    grid_max += bin_width * npad_bin
    grid_length += 2npad_bin

    # compute density
    k = KernelDensity.kde(
        x; npoints=grid_length, boundary=(grid_min, grid_max), bandwidth, kernel
    )
    grid = k.x
    pdf = k.density

    if bound_correction
        # reflect density at the boundary
        il = firstindex(pdf) + npad_bin
        ir = lastindex(pdf) - npad_bin
        pdf[il:(il + nbin_reflect)] .+= @view pdf[il:-1:(il - nbin_reflect)]
        pdf[(ir - nbin_reflect):ir] .+= @view pdf[(ir + nbin_reflect):-1:ir]
    end

    # remove padding
    grid_unpad = grid[(begin + npad_bin):(end - npad_bin)]
    pdf_unpad = pdf[(begin + npad_bin):(end - npad_bin)]

    return KernelDensity.UnivariateKDE(grid_unpad, pdf_unpad)
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
