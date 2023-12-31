"""
$(SIGNATURES)

Results of computing the widely applicable information criterion (WAIC).

See also: [`waic`](@ref), [`AbstractELPDResult`](@ref)

$(FIELDS)
"""
struct WAICResult{E,P} <: AbstractELPDResult
    "Estimates of the expected log pointwise predictive density (ELPD) and effective number of parameters (p)"
    estimates::E
    "Pointwise estimates"
    pointwise::P
end

function elpd_estimates(r::WAICResult; pointwise::Bool=false)
    return pointwise ? r.pointwise : r.estimates
end

function Base.show(io::IO, mime::MIME"text/plain", result::WAICResult; kwargs...)
    _show_elpd_estimates(io, mime, result; title="WAICResult with estimates", kwargs...)
    return nothing
end

"""
    waic(log_likelihood::AbstractArray) -> WAICResult{<:NamedTuple,<:NamedTuple}

Compute the widely applicable information criterion (WAIC).[^Watanabe2010][^Vehtari2017][^LOOFAQ]

`log_likelihood` must be an array of log-likelihood values with shape
`(chains, draws[, params...])`.

See also: [`WAICResult`](@ref), [`loo`](@ref)

[^Watanabe2010]: Watanabe, S. Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information Criterion in Singular Learning Theory. 11(116):3571−3594, 2010. https://jmlr.csail.mit.edu/papers/v11/watanabe10a.html
[^Vehtari2017]: Vehtari, A., Gelman, A. & Gabry, J.
    Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
    Stat Comput 27, 1413–1432 (2017).
    doi: [10.1007/s11222-016-9696-4](https://doi.org/10.1007/s11222-016-9696-4)
    arXiv: [1507.04544](https://arxiv.org/abs/1507.04544)
[^LOOFAQ]: Aki Vehtari. Cross-validation FAQ. https://mc-stan.org/loo/articles/online-only/faq.html

# Examples

Calculate WAIC of a model:

```jldoctest
julia> using ArviZExampleData

julia> idata = load_example_data("centered_eight");

julia> log_like = PermutedDimsArray(idata.log_likelihood.obs, (:draw, :chain, :school));

julia> waic(log_like)
WAICResult with estimates
 elpd  elpd_mcse    p  p_mcse
  -31        1.4  0.9    0.33
```
"""
waic(ll::AbstractArray) = _waic(ll)

function _waic(log_like, dims=(1, 2))
    _check_log_likelihood(log_like)

    # compute pointwise estimates
    lpd_i = _lpd_pointwise(log_like, dims)
    p_i = _maybe_scalar(dropdims(Statistics.var(log_like; corrected=true, dims); dims))
    elpd_i = lpd_i - p_i
    pointwise = (elpd=elpd_i, p=p_i)

    # combine estimates
    estimates = _elpd_estimates_from_pointwise(pointwise)

    return WAICResult(estimates, pointwise)
end
