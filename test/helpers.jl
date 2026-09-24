using BSON
using Distributions
using IntervalSets
using PosteriorStats
using Random
using StatsAPI: StatsAPI

function log_likelihood_eight_schools()
    dict = BSON.load(joinpath(@__DIR__, "data/eight_schools_loglikelihood.bson"))
    return (centered=dict[:centered], non_centered=dict[:non_centered])
end

function _isapprox(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}; kwargs...)
    return isapprox(collect(x), collect(y); kwargs...)
end
function _isapprox(x::AbstractInterval, y::AbstractInterval; kwargs...)
    return isleftclosed(x) == isleftclosed(y) &&
           isrightclosed(x) == isrightclosed(y) &&
           _isapprox(endpoints(x), endpoints(y); kwargs...)
end
function _isapprox(x::AbstractArray, y::AbstractArray; kwargs...)
    return all(map((x, y) -> _isapprox(x, y; kwargs...), x, y))
end
function _isapprox(x::Tuple, y::Tuple; kwargs...)
    length(x) == length(y) || return false
    return all(map((x, y) -> _isapprox(x, y; kwargs...), x, y))
end
function _isapprox(x::NamedTuple, y::NamedTuple; kwargs...)
    return keys(x) == keys(y) && _isapprox(values(x), values(y); kwargs...)
end
_isapprox(x, y; kwargs...) = isapprox(x, y; kwargs...)

# A conjugate normal model `yᵢ ~ Normal(μ, 1)` with `μ ~ Normal(0, 1)`, whose posterior given
# a subset of the observations is available in closed form, so that exact cross-validation
# estimates can be computed analytically.
function conjugate_normal_posterior(y, train_indices)
    precision = 1 + length(train_indices)
    return Normal(sum(y[train_indices]) / precision, 1 / sqrt(precision))
end

# an implementation of the Refitting interface from callables with the signatures of the
# interface methods, less the wrapper, for building test wrappers
struct CallableWrapper{R,L,J}
    nobs::Int
    refit::R
    loglikelihoods::L
    joint_loglikelihoods::J
end
function CallableWrapper(nobs; refit, loglikelihoods, joint_loglikelihoods=nothing)
    return CallableWrapper(nobs, refit, loglikelihoods, joint_loglikelihoods)
end
StatsAPI.nobs(wrapper::CallableWrapper) = wrapper.nobs
function PosteriorStats.refit(rng, wrapper::CallableWrapper, train_indices, eval_indices)
    return wrapper.refit(rng, train_indices, eval_indices)
end
function PosteriorStats.refit_loglikelihoods(
    rng, wrapper::CallableWrapper, fit, eval_indices
)
    return wrapper.loglikelihoods(rng, fit, eval_indices)
end
function PosteriorStats.refit_joint_loglikelihoods(
    rng, wrapper::CallableWrapper, fit, eval_indices
)
    wrapper.joint_loglikelihoods === nothing &&
        throw(ArgumentError("this wrapper has no joint log-likelihoods."))
    return wrapper.joint_loglikelihoods(rng, fit, eval_indices)
end

function conjugate_normal_wrapper(y; ndraws=1_000, nchains=4, joint=true)
    T = float(eltype(y))
    loglikelihoods =
        (_, mu, eval_indices) -> logpdf.(Normal.(mu), reshape(y[eval_indices], 1, 1, :))
    return CallableWrapper(
        length(y);
        refit=(rng, train_indices, _) ->
            T.(rand(rng, conjugate_normal_posterior(y, train_indices), ndraws, nchains)),
        loglikelihoods,
        # the observations are conditionally independent given `mu`
        joint_loglikelihoods=if joint
            (rng, mu, eval_indices) ->
                dropdims(sum(loglikelihoods(rng, mu, eval_indices); dims=3); dims=3)
        else
            nothing
        end,
    )
end

# a wrapper returning the closed-form predictive densities of the same model as a single
# draw, as the Refitting interface allows when `μ` can be marginalized analytically
function conjugate_normal_exact_wrapper(y)
    predictive(train_indices) = begin
        posterior = conjugate_normal_posterior(y, train_indices)
        return Normal(posterior.μ, hypot(1, posterior.σ)), posterior
    end
    return CallableWrapper(
        length(y);
        refit=(_, train_indices, _) -> predictive(train_indices),
        loglikelihoods=(_, (predictive, _), eval_indices) ->
            reshape(logpdf.(predictive, y[eval_indices]), 1, 1, :),
        joint_loglikelihoods=(_, (_, posterior), eval_indices) -> begin
            k = length(eval_indices)
            covariance = [posterior.σ^2 + (i == j) for i in 1:k, j in 1:k]
            return fill(
                logpdf(MvNormal(fill(posterior.μ, k), covariance), y[eval_indices]),
                1,
                1,
            )
        end,
    )
end

# exact `log p(yᵢ | y₋ᵢ)`, marginalizing over `μ` analytically. Here and below the
# observations are numbered `1:length(y)`, their column-major order, as the Refitting
# interface does
function conjugate_normal_exact_loo_elpd(y)
    indices = 1:length(y)
    elpd = similar(y, float(eltype(y)))
    for i in indices
        posterior = conjugate_normal_posterior(y, filter(!=(i), indices))
        elpd[i] = logpdf(Normal(posterior.μ, hypot(1, posterior.σ)), y[i])
    end
    return elpd
end

# exact `log p(yᵢ | y_train)` for the fold containing `i`, marginalizing over `μ` analytically
function conjugate_normal_exact_kfold_elpd(y, fold_ids)
    indices = 1:length(y)
    elpd = similar(y, float(eltype(y)))
    for label in unique(fold_ids)
        test_indices = findall(isequal(label), vec(fold_ids))
        posterior = conjugate_normal_posterior(y, setdiff(indices, test_indices))
        predictive = Normal(posterior.μ, hypot(1, posterior.σ))
        elpd[test_indices] .= logpdf.(predictive, y[test_indices])
    end
    return elpd
end

# exact joint `log p(y_E | y_T)` for each `(train_indices, test_indices)` split, marginalizing
# over `μ` analytically
function conjugate_normal_exact_joint_elpd(y, splits)
    return map(splits) do (train_indices, test_indices)
        posterior = conjugate_normal_posterior(y, train_indices)
        k = length(test_indices)
        covariance = [posterior.σ^2 + (i == j) for i in 1:k, j in 1:k]
        return logpdf(MvNormal(fill(posterior.μ, k), covariance), y[test_indices])
    end
end
