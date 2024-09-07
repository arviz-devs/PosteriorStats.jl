function eti(x::AbstractVecOrMat{<:Real}; prob::Real=DEFAULT_INTERVAL_PROB)
    0 < prob < 1 || throw(DomainError(prob, "ETI `prob` must be in the range `(0, 1)`."))
    isempty(x) && throw(ArgumentError("ETI cannot be computed for an empty array."))
    if any(isnan, x)
        T = float(promote_type(eltype(x), typeof(prob)))
        lower = upper = T(NaN)
    else
        alpha = (1 - prob) / 2
        lower, upper = Statistics.quantile(vec(x), (alpha, 1 - alpha))
    end
    return IntervalSets.ClosedInterval(lower, upper)
end
