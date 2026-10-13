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

