"""
$(TYPEDEF)

An abstract type representing the result of an ELPD computation.

Every subtype stores estimates of both the expected log predictive density (`elpd`) and the
effective number of parameters `p`, as well as standard errors and pointwise estimates of
each, from which other relevant estimates can be computed.

Subtypes implement the following functions:
- [`elpd_estimates`](@ref)
"""
abstract type AbstractELPDResult end

function _show_elpd_estimates(
    io::IO, mime::MIME"text/plain", r::AbstractELPDResult; kwargs...
)
    estimates = elpd_estimates(r)
    table = map(Base.vect, NamedTuple{(:elpd, :se_elpd, :p, :se_p)}(estimates))
    _show_prettytable(io, mime, table; kwargs...)
    return nothing
end

"""
    $(FUNCTIONNAME)(result::AbstractELPDResult; pointwise=false) -> (; elpd, se_elpd, lpd)

Return the (E)LPD estimates from the `result`.
"""
function elpd_estimates end

function _lpd_pointwise(log_likelihood, dims)
    ndraws = prod(Base.Fix1(size, log_likelihood), dims)
    lpd = LogExpFunctions.logsumexp(log_likelihood; dims)
    T = eltype(lpd)
    return dropdims(lpd; dims) .- log(T(ndraws))
end

function _elpd_estimates_from_pointwise(pointwise)
    elpd, se_elpd = _sum_and_se(pointwise.elpd)
    p, se_p = _sum_and_se(pointwise.p)
    return (; elpd, se_elpd, p, se_p)
end
