function default_ci_fun()
    ci_kind = Symbol(Preferences.load_preference(PosteriorStats, "ci_kind", "eti"))
    ci_kind ∈ (:eti, :hdi) ||
        throw(ArgumentError("Invalid ci_kind: $ci_kind. Must be one of (eti, hdi)."))
    return ci_kind == :eti ? eti : hdi
end

function default_ci_prob((::Type{T})=Float64) where {T<:Real}
    prob = T(Preferences.load_preference(PosteriorStats, "ci_prob", 0.94))
    0 < prob < 1 || throw(DomainError(prob, "ci_prob must be in the range (0, 1)."))
    return prob
end

