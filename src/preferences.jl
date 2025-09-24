function default_ci_fun()
    ci_kind = Symbol(Preferences.load_preference(PosteriorStats, "ci_kind", "eti"))
    ci_kind ∈ (:eti, :hdi) ||
        throw(ArgumentError("Invalid ci_kind: $ci_kind. Must be one of (eti, hdi)."))
    return ci_kind == :eti ? eti : hdi
end

function default_ci_prob((::Type{T})=Float32) where {T<:Real}
    prob = T(Preferences.load_preference(PosteriorStats, "ci_prob", 0.89))
    0 < prob < 1 || throw(DomainError(prob, "ci_prob must be in the range (0, 1)."))
    return prob
end

function default_point_estimate()
    point_estimate = Symbol(
        Preferences.load_preference(PosteriorStats, "point_estimate", "mean")
    )
    if point_estimate === :mean
        return Statistics.mean
    elseif point_estimate === :median
        return Statistics.median
    elseif point_estimate === :mode
        return StatsBase.mode
    else
        throw(
            ArgumentError(
                "Invalid point_estimate: $point_estimate. Must be one of (mean, median, mode).",
            ),
        )
    end
end

function default_weights_method()
    method = Preferences.load_preference(PosteriorStats, "weights_method", "Stacking")
    method == "Stacking" && return Stacking
    method == "PseudoBMA" && return PseudoBMA
    method == "BootstrappedPseudoBMA" && return BootstrappedPseudoBMA
    throw(
        ArgumentError(
            "Invalid weights_method: $method. Must be one of " *
            "(Stacking, PseudoBMA, BootstrappedPseudoBMA).",
        ),
    )
end

@kwdef struct PrecisionSettings
    show_printf::String = ""
    show_sigdigits_default::Int = 3
    show_sigdigits_se::Int = 2
    show_sigdigits_rhat::Int = 2
    show_sigdigits_using_se::Bool = true
    show_ess_int::Bool = true
    show_html_max_rows::Int = 25
end

function default_precision_settings()
    default_settings = PrecisionSettings()
    # load settings from preferences
    settings = PrecisionSettings(
        (
            _parse(
                T,
                Preferences.load_preference(
                    PosteriorStats, string(k), getproperty(default_settings, k)
                ),
            ) for
            (k, T) in zip(fieldnames(PrecisionSettings), fieldtypes(PrecisionSettings))
        )...,
    )
    # validate settings
    for (k, T) in zip(fieldnames(PrecisionSettings), fieldtypes(PrecisionSettings))
        if T <: Int
            v = getfield(settings, k)
            v ≥ 0 || throw(DomainError(v, "Setting `$k` must be non-negative"))
        end
    end
    return settings
end

_parse(::Type{T}, v::T) where {T} = v
_parse(::Type{T}, v) where {T} = parse(T, v)
