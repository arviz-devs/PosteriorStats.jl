# Density estimation methods and utilities

"""
    DensityEstimationMethod

Abstract type for density estimation methods.

Densities are defined wrt the Lebesgue measure on subintervals of the real line for
continuous data or the counting measure on unique values for discrete data.

Each method should implement:
- `bins_and_probs(::DensityEstimationMethod, x::AbstractVector{<:Real}) -> (bins, probs)`
    Returns the density evaluated at regularly spaced points, normalized to sum to 1.
    Empty bins may be omitted.
- `density_at(::DensityEstimationMethod, x::AbstractVector{<:Real}) -> densities`
    Returns the density evaluated at the points in `x`
"""
abstract type DensityEstimationMethod end

"""
    DefaultDensityEstimation <: DensityEstimationMethod

Select density estimation method based on data type.
"""
struct DefaultDensityEstimation <: DensityEstimationMethod end

"""
    DiscreteDensityEstimation <: DensityEstimationMethod

Estimate density for integer-valued data using the counting measure.
"""
struct DiscreteDensityEstimation <: DensityEstimationMethod end

function bins_and_probs(::DiscreteDensityEstimation, x::AbstractVector{<:Real})
    prop_map = OrderedCollections.OrderedDict(StatsBase.proportionmap(x))
    sort!(prop_map)
    bins = collect(keys(prop_map))
    probs = collect(values(prop_map))
    return bins, probs
end

"""
    HistogramEstimation{K} <: DensityEstimationMethod

Estimate piecewise constant density using a histogram.
"""
struct HistogramEstimation{K} <: DensityEstimationMethod
    hist_kwargs::K
end
HistogramEstimation(; hist_kwargs...) = HistogramEstimation(hist_kwargs)

function bins_and_probs(est::HistogramEstimation, x::AbstractVector{<:Real})
    hist = StatsBase.fit(StatsBase.Histogram, x; est.hist_kwargs...)
    return StatsBase.midpoints(hist.edges[1]), normalize(hist; mode=:probability).weights
end

function density_at(est::HistogramEstimation, x::AbstractVector{<:Real})
    hist = normalize(
        StatsBase.fit(StatsBase.Histogram, x; est.hist_kwargs...); mode=:density
    )
    return _histogram_density.(Ref(hist), x)
end

function _histogram_density(hist::StatsBase.Histogram, x::Real)
    edges = only(hist.edges)
    bin_index = _binindex(edges, hist.closed, x)
    weights = hist.weights
    return get(weights, bin_index, zero(eltype(weights)))
end

function _binindex(edges::AbstractVector, closed::Symbol, x::Real)
    if closed === :right
        return searchsortedfirst(edges, x) - 1
    else
        return searchsortedlast(edges, x)
    end
end
