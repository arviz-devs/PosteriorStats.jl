function default_ci_fun()
    ci_kind = Symbol(Preferences.load_preference(PosteriorStats, "ci_kind", "eti"))
    ci_kind ∈ (:eti, :hdi) ||
        throw(ArgumentError("Invalid ci_kind: $ci_kind. Must be one of (eti, hdi)."))
    return ci_kind == :eti ? eti : hdi
end

