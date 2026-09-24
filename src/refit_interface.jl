"""
    refit(rng::Random.AbstractRNG, wrapper, train_indices, eval_indices) -> fit

Fit the wrapped model to the observations `train_indices`, to be scored at `eval_indices`.

Part of the [Refitting interface](@ref refitting). Both index arguments are vectors of
integers in `1:nobs`. The fit must represent the posterior given the observations
`train_indices` only; `eval_indices` are the observations it will be scored at, which the
wrapper may use to include their latent variables in the fit, or ignore. The returned `fit`
is opaque and is only passed back to the log-likelihood methods with `eval_indices`. All
randomness should be drawn from `rng`, which is seeded per refit so that the results are
reproducible whether the refits run sequentially or in parallel.
"""
function refit end

"""
    refit_loglikelihoods(rng, wrapper, fit, eval_indices) -> AbstractArray{<:Real,3}

Pointwise log-likelihoods of the observations `eval_indices` at the draws in `fit`.

Part of the [Refitting interface](@ref refitting). The result has shape
`(draws, chains, length(eval_indices))`, in the order requested, and entry ``i`` is
``\\log p(y_i \\mid \\theta, y_T)``, conditional on the observations ``T`` the fit was
trained on and marginal over the other observations in `eval_indices`. A predictive density
available in closed form is returned as a single draw, with shape
`(1, 1, length(eval_indices))`. `rng` is the same seeded `Random.AbstractRNG` that
[`PosteriorStats.refit`](@ref) received, for any simulation the marginalization needs.
"""
function refit_loglikelihoods end

"""
    refit_joint_loglikelihoods(rng, wrapper, fit, eval_indices) -> AbstractMatrix{<:Real}

Joint log-likelihood of the observations `eval_indices` at the draws in `fit`.

Optional part of the [Refitting interface](@ref refitting), needed only to score a set of
observations as one predictive event. The result has shape `(draws, chains)` and holds
``\\log p(y_E \\mid \\theta, y_T)``, conditional on the observations ``T`` the fit was
trained on, or a single entry if available in closed form. It is not derived by summing
[`PosteriorStats.refit_loglikelihoods`](@ref), which gives the joint only for conditionally
independent observations. `rng` is as for [`PosteriorStats.refit_loglikelihoods`](@ref).
"""
function refit_joint_loglikelihoods end

"""
    slice_axes(cells::AbstractVector{<:CartesianIndex}) -> Tuple

Per-dimension index vectors whose product is the set of `cells`.

For a wrapper whose model needs a rectangular block of an observation array `y`,
`y[slice_axes(CartesianIndices(y)[train_indices])...]` selects the block held in
`train_indices`. Throws an `ArgumentError` if the cells do not form a block, as when
[`reloo`](@ref) holds out single cells.

# Examples

```jldoctest
julia> y = reshape(1:12, 3, 4);

julia> PosteriorStats.slice_axes(CartesianIndices(y)[vec(LinearIndices(y)[:, [1, 3]])])
([1, 2, 3], [1, 3])
```
"""
function slice_axes(cells::AbstractArray{CartesianIndex{M}}) where {M}
    slice = ntuple(d -> sort!(unique(getindex.(cells, d))), M)
    if !allunique(cells) || prod(length, slice) != length(cells)
        throw(
            ArgumentError(
                "the observation indices do not form a rectangular block of the" *
                " observation array.",
            ),
        )
    end
    return slice
end

# the cell of each observation in an array of the observations' shape, so that `cells[i]`
# is the position of observation `i` in its column-major numbering, whatever the axes of `x`
_observation_cells(x::AbstractArray) = vec(collect(CartesianIndices(x)))

# Helpers shared by the statistics built on the interface

function _check_nobs(wrapper, nobservations, source)
    nobs_wrapper = StatsAPI.nobs(wrapper)
    if nobs_wrapper != nobservations
        throw(
            DimensionMismatch(
                "`wrapper` has $nobs_wrapper observations, but `$source` has" *
                " $nobservations.",
            ),
        )
    end
    return nothing
end

# the interface methods, with their results validated
function _loglikelihoods(rng, wrapper, fit, eval_indices)
    log_like = refit_loglikelihoods(rng, wrapper, fit, eval_indices)
    _check_loglikelihoods(log_like, eval_indices)
    return log_like
end
function _joint_loglikelihoods(rng, wrapper, fit, eval_indices)
    log_like = refit_joint_loglikelihoods(rng, wrapper, fit, eval_indices)
    _check_joint_loglikelihoods(log_like, eval_indices)
    return log_like
end

function _check_loglikelihoods(log_like, eval_indices)
    if ndims(log_like) != 3
        throw(
            DimensionMismatch(
                "`PosteriorStats.refit_loglikelihoods` must return a 3-dimensional array" *
                " with shape `(draws, chains, length(eval_indices))`, but it returned a" *
                " $(ndims(log_like))-dimensional array. Reshape the array if the draws" *
                " were not drawn in multiple chains.",
            ),
        )
    end
    nvals = size(log_like, 3)
    if nvals != length(eval_indices)
        throw(
            DimensionMismatch(
                "`PosteriorStats.refit_loglikelihoods` must return one log-likelihood" *
                " value per entry of `eval_indices`, but it returned $nvals values for" *
                " $(length(eval_indices)) indices $(eval_indices).",
            ),
        )
    end
    return nothing
end

function _check_joint_loglikelihoods(log_like, eval_indices)
    if ndims(log_like) != 2
        throw(
            DimensionMismatch(
                "`PosteriorStats.refit_joint_loglikelihoods` must return a 2-dimensional" *
                " array with shape `(draws, chains)`, but it returned a" *
                " $(ndims(log_like))-dimensional array for $(length(eval_indices))" *
                " observations. It must return one value per draw, not one per" *
                " observation.",
            ),
        )
    end
    return nothing
end

# refit on `train_indices` and estimate the ELPD of each observation in `eval_indices`
function _refit_pointwise_elpd(rng, wrapper, train_indices, eval_indices)
    fit = refit(rng, wrapper, train_indices, eval_indices)
    return fit, _exact_elpd_pointwise(_loglikelihoods(rng, wrapper, fit, eval_indices))
end

# refit on `train_indices` and estimate the joint ELPD of the observations in `eval_indices`
function _refit_joint_elpd(rng, wrapper, train_indices, eval_indices)
    fit = refit(rng, wrapper, train_indices, eval_indices)
    return fit, _exact_elpd_joint(_joint_loglikelihoods(rng, wrapper, fit, eval_indices))
end

function _check_ntasks(ntasks::Int)
    ntasks ≥ 1 || throw(ArgumentError("`ntasks` must be at least 1, got $ntasks."))
    return nothing
end

# `map(f, xs)` where `f(rng_x, x)` receives a copy of `rng` seeded with a seed drawn from
# `rng` for `x` up front, as AbstractMCMC seeds its chains, running at most `ntasks` calls
# concurrently. The results depend on neither `ntasks` nor the order the calls run in.
function _map_seeded(f, rng::Random.AbstractRNG, xs::AbstractVector, ntasks::Int)
    seeds = rand(rng, UInt, length(xs))
    call(x, seed) = f(Random.seed!(copy(rng), seed), x)
    ntasks == 1 && return map(call, xs, seeds)
    n = length(xs)
    results = Vector{Any}(undef, n)
    next = Threads.Atomic{Int}(1)
    tasks = map(1:min(ntasks, n)) do _
        # each task takes the next unprocessed element until none are left, so the work is
        # balanced even when the calls take different times
        return Threads.@spawn try
            while (k = Threads.atomic_add!(next, 1)) ≤ n
                results[k] = call(xs[k], seeds[k])
            end
        catch
            # stop the other tasks from taking further elements after a failure
            next[] = n + 1
            rethrow()
        end
    end
    foreach(wait, tasks)
    return map(identity, results)
end

# exact pointwise ELPD estimates from log-likelihood values evaluated at draws from the
# corresponding cross-validation posteriors
function _exact_elpd_pointwise(log_like::AbstractArray{<:Real,3})
    dims = (1, 2)
    ndraws = prod(Base.Fix1(size, log_like), dims)
    T = typeof(float(one(eltype(log_like))))
    if ndraws == 1
        # a single draw is a predictive density known in closed form, without Monte Carlo
        # error (see the Refitting interface)
        elpd = dropdims(T.(log_like); dims)
        return (elpd, se_elpd=zero(elpd), reff=one.(elpd))
    end
    log_weights = -log(T(ndraws))
    elpd = _log_mean(log_like, log_weights; dims)
    like = LogExpFunctions.softmax(log_like; dims)
    reff = MCMCDiagnosticTools.ess(like; kind=:basic, split_chains=1, relative=true)
    se_elpd = _se_log_mean(log_like, log_weights; dims, log_mean=elpd)
    return (;
        elpd=dropdims(elpd; dims), se_elpd=dropdims(se_elpd; dims) ./ sqrt.(reff), reff
    )
end

# the same for a set of observations scored as one predictive event, which is a single
# observation to the pointwise machinery
function _exact_elpd_joint(log_like::AbstractMatrix{<:Real})
    return map(only, _exact_elpd_pointwise(reshape(log_like, size(log_like)..., 1)))
end
