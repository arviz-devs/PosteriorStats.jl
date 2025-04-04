const DEFAULT_STACKING_OPTIMIZER = Optim.LBFGS()

"""
$(TYPEDEF)

An abstract type representing methods for computing model weights.

Subtypes implement [`model_weights`](@ref)`(method, elpd_results)`.
"""
abstract type AbstractModelWeightsMethod end

"""
    model_weights(elpd_results; method=Stacking())
    model_weights(method::AbstractModelWeightsMethod, elpd_results)

Compute weights for each model in `elpd_results` using `method`.

`elpd_results` is a `Tuple`, `NamedTuple`, or `AbstractVector` with
[`AbstractELPDResult`](@ref) entries. The weights are returned in the same type of
collection.

[`Stacking`](@ref) is the recommended approach, as it performs well even when the true data
generating process is not included among the candidate models. See [Yao2018](@citet) for
details.

See also: [`AbstractModelWeightsMethod`](@ref), [`compare`](@ref)

# Examples

Compute [`Stacking`](@ref) weights for two models:

```jldoctest model_weights; filter = [r"└.*", r"(\\d+\\.\\d{3})\\d*" => s"\\1"]
julia> using ArviZExampleData

julia> models = (
           centered=load_example_data("centered_eight"),
           non_centered=load_example_data("non_centered_eight"),
       );

julia> elpd_results = map(models) do idata
           log_like = PermutedDimsArray(idata.log_likelihood.obs, (2, 3, 1))
           return loo(log_like)
       end;
┌ Warning: 1 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.
└ @ PSIS ~/.julia/packages/PSIS/...

julia> model_weights(elpd_results; method=Stacking()) |> pairs
pairs(::NamedTuple) with 2 entries:
  :centered     => 3.50546e-31
  :non_centered => 1.0
```

Now we compute [`BootstrappedPseudoBMA`](@ref) weights for the same models:

```jldoctest model_weights; setup = :(using Random; Random.seed!(94))
julia> model_weights(elpd_results; method=BootstrappedPseudoBMA()) |> pairs
pairs(::NamedTuple) with 2 entries:
  :centered     => 0.492513
  :non_centered => 0.507487
```

# References

- [Yao2018](@cite) Yao et al. Bayesian Analysis 13, 3 (2018)
"""
function model_weights(elpd_results; method::AbstractModelWeightsMethod=Stacking())
    return model_weights(method, elpd_results)
end

# Akaike-type weights are defined as exp(-AIC/2), normalized to 1, which on the log-score
# IC scale is equivalent to softmax
akaike_weights!(w, elpds) = LogExpFunctions.softmax!(w, elpds)
_akaike_weights(elpds) = _softmax(elpds)

"""
$(TYPEDEF)

Model weighting method using pseudo Bayesian Model Averaging (pseudo-BMA) and Akaike-type
weighting.

    PseudoBMA(; regularize=false)
    PseudoBMA(regularize)

Construct the method with optional regularization of the weights using the standard error of
the ELPD estimate.

!!! note

    This approach is not recommended, as it produces unstable weight estimates. It is
    recommended to instead use [`BootstrappedPseudoBMA`](@ref) to stabilize the weights
    or [`Stacking`](@ref). For details, see [Yao2018](@citet).

See also: [`Stacking`](@ref)

# References

- [Yao2018](@cite) Yao et al. Bayesian Analysis 13, 3 (2018)
"""
struct PseudoBMA <: AbstractModelWeightsMethod
    regularize::Bool
end
PseudoBMA(; regularize::Bool=false) = PseudoBMA(regularize)

function model_weights(method::PseudoBMA, elpd_results)
    elpds = map(elpd_results) do result
        est = elpd_estimates(result)
        method.regularize || return est.elpd
        return est.elpd - est.se_elpd / 2
    end
    return _akaike_weights(elpds)
end

"""
$(TYPEDEF)

Model weighting method using pseudo Bayesian Model Averaging using Akaike-type weighting
with the Bayesian bootstrap (pseudo-BMA+)[Yao2018](@citep).

The Bayesian bootstrap stabilizes the model weights.

    BootstrappedPseudoBMA(; rng=Random.default_rng(), samples=1_000, alpha=1)
    BootstrappedPseudoBMA(rng, samples, alpha)

Construct the method.

$(TYPEDFIELDS)

See also: [`Stacking`](@ref)

# References

- [Yao2018](@cite) Yao et al. Bayesian Analysis 13, 3 (2018)
"""
struct BootstrappedPseudoBMA{R<:Random.AbstractRNG,T<:Real} <: AbstractModelWeightsMethod
    "The random number generator to use for the Bayesian bootstrap"
    rng::R
    "The number of samples to draw for bootstrapping"
    samples::Int
    """The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap.
    The default (1) corresponds to a uniform distribution on the simplex."""
    alpha::T
end
function BootstrappedPseudoBMA(;
    rng::Random.AbstractRNG=Random.default_rng(), samples::Int=1_000, alpha::Real=1
)
    return BootstrappedPseudoBMA(rng, samples, alpha)
end

function model_weights(method::BootstrappedPseudoBMA, elpd_results)
    _elpd = vec(elpd_estimates(first(values(elpd_results)); pointwise=true).elpd)
    α = similar(_elpd)
    n = length(α)
    rng = method.rng
    α_dist = Distributions.Dirichlet(n, method.alpha)
    ic_mat = _elpd_matrix(elpd_results)
    elpd_mean = similar(ic_mat, axes(ic_mat, 2))
    weights_mean = zero(elpd_mean)
    w = similar(weights_mean)
    for _ in 1:(method.samples)
        _model_weights_bootstrap!(w, elpd_mean, α, rng, α_dist, ic_mat)
        weights_mean .+= w
    end
    weights_mean ./= method.samples
    return _assimilar(elpd_results, weights_mean)
end

function _model_weights_bootstrap!(w, elpd_mean, α, rng, α_dist, ic_mat)
    Random.rand!(rng, α_dist, α)
    mul!(elpd_mean, ic_mat', α)
    elpd_mean .*= length(α)
    akaike_weights!(w, elpd_mean)
    return w
end

"""
$(TYPEDEF)

Model weighting using stacking of predictive distributions[Yao2018](@citep).

    Stacking(; optimizer=Optim.LBFGS(), options=Optim.Options()
    Stacking(optimizer[, options])

Construct the method, optionally customizing the optimization.

$(TYPEDFIELDS)

See also: [`BootstrappedPseudoBMA`](@ref)

# References

- [Yao2018](@cite) Yao et al. Bayesian Analysis 13, 3 (2018)
"""
struct Stacking{O<:Optim.AbstractOptimizer} <: AbstractModelWeightsMethod
    """The optimizer to use for the optimization of the weights. The optimizer must support
    projected gradient optimization via a `manifold` field."""
    optimizer::O
    """The Optim options to use for the optimization of the weights."""
    options::Optim.Options

    function Stacking(
        optimizer::Optim.AbstractOptimizer, options::Optim.Options=Optim.Options()
    )
        hasfield(typeof(optimizer), :manifold) ||
            throw(ArgumentError("The optimizer must have a `manifold` field."))
        _optimizer = Setfield.@set optimizer.manifold = Optim.Sphere()
        return new{typeof(_optimizer)}(_optimizer, options)
    end
end
function Stacking(;
    optimizer::Optim.AbstractOptimizer=DEFAULT_STACKING_OPTIMIZER,
    options::Optim.Options=Optim.Options(),
)
    return Stacking(optimizer, options)
end

function model_weights(method::Stacking, elpd_pairs)
    ic_mat = _elpd_matrix(elpd_pairs)
    exp_ic_mat = exp.(ic_mat)
    _, weights = _model_weights_stacking(exp_ic_mat, method.optimizer, method.options)
    return _assimilar(elpd_pairs, weights)
end
function _model_weights_stacking(exp_ic_mat, optimizer, options)
    # set up optimization objective
    objective = InplaceStackingOptimObjective(exp_ic_mat)

    # set up initial point on optimization manifold
    w0 = similar(exp_ic_mat, axes(exp_ic_mat, 2))
    fill!(w0, 1//length(w0))
    x0 = _initial_point(objective, w0)

    # optimize
    sol = Optim.optimize(Optim.only_fg!(objective), x0, optimizer, options)

    # check convergence
    Optim.converged(sol) ||
        @warn "Optimization of stacking weights failed to converge after $(Optim.iterations(sol)) iterations."

    # return solution and weights
    w = _final_point(objective, sol.minimizer)
    return sol, w
end

function _elpd_matrix(elpd_results)
    elpd_values = map(elpd_results) do result
        return vec(elpd_estimates(result; pointwise=true).elpd)
    end
    return reduce(hcat, collect(elpd_values))
end

# Optimize on the probability simplex by converting the problem to optimization on the unit
# sphere, optimizing with projected gradients, and mapping the solution back to the sphere.
# When the objective function on the simplex is convex, each global minimizer on the sphere
# maps to the global minimizer on the simplex, but the optimization manifold is simple, and
# no inequality constraints exist.
# Q Li, D McKenzie, W Yin. "From the simplex to the sphere: faster constrained optimization
# using the Hadamard parametrization." Inf. Inference. 12.3 (2023): iaad017.
# doi: 10.1093/imaiai/iaad017. arXiv: 2112.05273
struct InplaceStackingOptimObjective{E,C}
    exp_ic_mat::E
    cache::C
end
function InplaceStackingOptimObjective(exp_ic_mat)
    cache = (
        similar(exp_ic_mat, axes(exp_ic_mat, 1)), similar(exp_ic_mat, axes(exp_ic_mat, 2))
    )
    return InplaceStackingOptimObjective(exp_ic_mat, cache)
end
function (obj::InplaceStackingOptimObjective)(F, G, x)
    exp_ic_mat = obj.exp_ic_mat
    cache, w = obj.cache
    _sphere_to_simplex!(w, x)
    mul!(cache, exp_ic_mat, w)
    cache .= inv.(cache)
    if G !== nothing
        mul!(G, exp_ic_mat', cache)
        G .*= -1
        _∇sphere_to_simplex!(G, x)
    end
    if F !== nothing
        return sum(log, cache)
    end
    return nothing
end
_initial_point(::InplaceStackingOptimObjective, w0) = _simplex_to_sphere(w0)
_final_point(::InplaceStackingOptimObjective, x) = _sphere_to_simplex(x)

# if ∑xᵢ² = 1, then if wᵢ = xᵢ², then w is on the probability simplex
_sphere_to_simplex(x) = x .^ 2
function _sphere_to_simplex!(w, x)
    w .= x .^ 2
    return w
end
_simplex_to_sphere(x) = sqrt.(x)
function _∇sphere_to_simplex!(∂x, x)
    ∂x .*= 2 .* x
    return ∂x
end
