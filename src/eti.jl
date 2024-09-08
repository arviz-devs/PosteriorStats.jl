function eti(
    x::AbstractArray{<:Real};
    prob::Real=DEFAULT_INTERVAL_PROB,
    sorted::Bool=false,
    kwargs...,
)
    return eti!(sorted ? x : _copymutable(x); prob, sorted, kwargs...)
end

function eti!(x::AbstractArray{<:Real}; prob::Real=DEFAULT_INTERVAL_PROB, kwargs...)
    ndims(x) > 0 ||
        throw(ArgumentError("ETI cannot be computed for a 0-dimensional array."))
    0 < prob < 1 || throw(DomainError(prob, "ETI `prob` must be in the range `(0, 1)`."))
    isempty(x) && throw(ArgumentError("ETI cannot be computed for an empty array."))
    return _eti!(x, prob; kwargs...)
end

function _eti!(x::AbstractVecOrMat{<:Real}, prob::Real; kwargs...)
    if any(isnan, x)
        T = float(promote_type(eltype(x), typeof(prob)))
        lower = upper = T(NaN)
    else
        alpha = (1 - prob) / 2
        lower, upper = Statistics.quantile!(vec(x), (alpha, 1 - alpha); kwargs...)
    end
    return IntervalSets.ClosedInterval(lower, upper)
end
function _eti!(x::AbstractArray, prob::Real; kwargs...)
    axes_out = _param_axes(x)
    T = float(promote_type(eltype(x), typeof(prob)))
    interval = similar(x, IntervalSets.ClosedInterval{T}, axes_out)
    for (i, x_slice) in zip(eachindex(interval), _eachparam(x))
        interval[i] = _eti!(x_slice, prob; kwargs...)
    end
    return interval
end
