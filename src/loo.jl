"""
$(SIGNATURES)

Results of Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

See also: [`loo`](@ref), [`AbstractELPDResult`](@ref)

$(FIELDS)
"""
struct PSISLOOResult{E,P,R<:PSIS.PSISResult} <: AbstractELPDResult
    "Estimates of the expected log pointwise predictive density (ELPD) and effective number of parameters (p)"
    estimates::E
    "Pointwise estimates"
    pointwise::P
    "A [`PSIS.PSISResult`](@extref) with Pareto-smoothed importance sampling (PSIS) results"
    psis_result::R
end

function elpd_estimates(r::PSISLOOResult; pointwise::Bool=false)
    return pointwise ? r.pointwise : r.estimates
end

function Base.show(io::IO, mime::MIME"text/plain", result::PSISLOOResult; kwargs...)
    _show_elpd_estimates(io, mime, result; title="PSISLOOResult with estimates", kwargs...)
    println(io)
    println(io)
    print(io, "and ")
    show(io, mime, result.psis_result)
    return nothing
end

"""
    loo(log_likelihood; reff=nothing, kwargs...) -> PSISLOOResult{<:NamedTuple,<:NamedTuple}

Compute the Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).
[Vehtari2017, LOOFAQ](@cite)

`log_likelihood` must be an array of log-likelihood values with shape
`(chains, draws[, params...])`.

# Keywords

  - `reff::Union{Real,AbstractArray{<:Real}}`: The relative effective sample size(s) of the
    _likelihood_ values. If an array, it must have the same data dimensions as the
    corresponding log-likelihood variable. If not provided, then this is estimated using
    [`MCMCDiagnosticTools.ess`](@extref).
  - `kwargs`: Remaining keywords are forwarded to [`PSIS.psis`](@extref).

See also: [`PSISLOOResult`](@ref), [`waic`](@ref)

# Examples

Manually compute ``R_\\mathrm{eff}`` and calculate PSIS-LOO of a model:

```jldoctest
julia> using ArviZExampleData, MCMCDiagnosticTools

julia> idata = load_example_data("centered_eight");

julia> log_like = PermutedDimsArray(idata.log_likelihood.obs, (:draw, :chain, :school));

julia> reff = ess(log_like; kind=:basic, split_chains=1, relative=true);

julia> loo(log_like; reff)
PSISLOOResult with estimates
 elpd  elpd_mcse    p  p_mcse
  -31        1.4  0.9    0.33

and PSISResult with 500 draws, 4 chains, and 8 parameters
Pareto shape (k) diagnostic values:
                    Count      Min. ESS
 (-Inf, 0.5]  good  5 (62.5%)  290
  (0.5, 0.7]  okay  3 (37.5%)  399
```

# References

- [Vehtari2017](@cite) Vehtari et al. Stat. Comput. 27 (2017).
- [LOOFAQ](@cite) Vehtari. Cross-validation FAQ.
"""
loo(ll::AbstractArray; kwargs...) = _loo(ll; kwargs...)

function _psis_loo_setup(log_like, _reff; kwargs...)
    if _reff === nothing
        # normalize log likelihoods to improve numerical stability of ESS estimate
        like = LogExpFunctions.softmax(log_like; dims=(1, 2))
        reff = MCMCDiagnosticTools.ess(like; kind=:basic, split_chains=1, relative=true)
    else
        reff = _reff
    end
    # smooth importance weights
    psis_result = PSIS.psis(-log_like, reff; kwargs...)
    return psis_result
end

function _loo(log_like; reff=nothing, kwargs...)
    _check_log_likelihood(log_like)
    psis_result = _psis_loo_setup(log_like, reff; kwargs...)
    return _loo(log_like, psis_result)
end
function _loo(log_like, psis_result, dims=(1, 2))
    # compute pointwise estimates
    lpd_i = _maybe_scalar(_lpd_pointwise(log_like, dims))
    elpd_i, se_elpd_i = map(
        _maybe_scalar, _elpd_loo_pointwise_and_se(psis_result, log_like, dims)
    )
    p_i = lpd_i - elpd_i
    pointwise = (;
        elpd=elpd_i,
        se_elpd=se_elpd_i,
        p=p_i,
        reff=psis_result.reff,
        pareto_shape=psis_result.pareto_shape,
    )

    # combine estimates
    estimates = _elpd_estimates_from_pointwise(pointwise)

    return PSISLOOResult(estimates, pointwise, psis_result)
end

function _elpd_loo_pointwise_and_se(psis_result::PSIS.PSISResult, log_likelihood, dims)
    log_norm = LogExpFunctions.logsumexp(psis_result.log_weights; dims)
    log_weights = psis_result.log_weights .- log_norm
    elpd_i = _log_mean(log_likelihood, log_weights; dims)
    elpd_i_se = _se_log_mean(log_likelihood, log_weights; dims, log_mean=elpd_i)
    return (
        elpd=_maybe_scalar(dropdims(elpd_i; dims)),
        se_elpd=_maybe_scalar(dropdims(elpd_i_se; dims) ./ sqrt.(psis_result.reff)),
    )
end
