"""
$(SIGNATURES)

Results of cross-validation by refitting, from [`cross_validate`](@ref).

See also: [`AbstractELPDResult`](@ref)

$(FIELDS)
"""
struct CrossValidationELPDResult{E,P,S,T} <: AbstractELPDResult
    """Estimates of the ELPD and effective number of parameters `p`, the latter `missing`
    without `log_likelihood`"""
    estimates::E
    """Pointwise estimates by observation (`joint=false`) or by split; `missing` for
    observations held out in no split"""
    pointwise::P
    "`(; train_indices, test_indices)` of each split, in the order they were fit"
    splits::S
    "The fit of each split if `save_fits=true`, otherwise `nothing`"
    fits::T
    "`:observation` or `:split`, the predictive unit the estimates refer to"
    estimand::Symbol
    "The normalization of the scores, `:observation`, `:split` or `nothing`"
    normalize_by::Union{Nothing,Symbol}
end

_elpd_estimand(r::CrossValidationELPDResult) = r.estimand

function elpd_estimates(r::CrossValidationELPDResult; pointwise::Bool=false)
    return pointwise ? r.pointwise : r.estimates
end

function Base.show(
    io::IO, mime::MIME"text/plain", result::CrossValidationELPDResult; kwargs...
)
    _show_elpd_estimates(
        io, mime, result; title="CrossValidationELPDResult with estimates", kwargs...
    )
    println(io)
    println(io)
    print(io, "computed from ", _describe_splits(result))
    return nothing
end

function _describe_splits(result::CrossValidationELPDResult)
    nsplits = length(result.splits)
    if result.estimand === :split
        lo, hi = extrema(split -> length(split.test_indices), result.splits)
        sizes = lo == hi ? "$lo" : "$lo to $hi"
        description = "$nsplits splits of $sizes held-out observations each, scored jointly"
        result.normalize_by === :split && (description *= ", normalized by split")
        return description
    end
    nobservations = length(elpd_estimates(result; pointwise=true).elpd)
    coverage = _coverage(result.splits, nobservations)
    ntested = count(>(0), coverage)
    cmax = maximum(coverage)
    if ntested == nobservations && cmax == 1 && result.normalize_by === :observation
        if all(split -> length(split.train_indices) == nobservations - 1, result.splits)
            return "leave-one-out cross-validation of $nobservations observations"
        end
        return "$nsplits-fold cross-validation of $nobservations observations"
    end
    description = "$nsplits splits covering $ntested of $nobservations observations"
    cmax > 1 && (description *= ", up to $cmax times each")
    result.normalize_by === :split && (description *= ", normalized by split")
    result.normalize_by === nothing && (description *= ", summing the scores")
    return description
end

"""
    cross_validate([rng,] wrapper[, log_likelihood]; kwargs...) -> CrossValidationELPDResult

Estimate the ELPD by cross-validation, refitting the model once per split.
[Vehtari2017, LOOFAQ](@cite)

# Arguments

  - `rng::Random.AbstractRNG=Random.default_rng()`: draws the random folds and a seed for
    each split, from which the split's refit is given a seeded copy of `rng`, so that the
    results are reproducible for any `ntasks`.
  - `log_likelihood`: the pointwise log-likelihood of a fit to all observations, with shape
    `(draws, chains, observations...)` as for [`loo`](@ref), or the exact in-sample
    predictive densities as a single draw. Used only to compute the effective number of
    parameters `p`, which is otherwise `missing`, and accepted only when every split trains
    on all the observations it does not hold out; see
    [Cross-validation by refitting](@ref cross_validation) for what it assumes. If given,
    the pointwise estimates have the shape of the observations; otherwise they are a vector.

# Keywords

  - `folds=10`: the splits, as an integer ``K`` (random folds drawn with `rng`), an array of
    fold labels of any type but tuples, in the shape or column-major order of the
    observations, or a vector of `(train, test)` tuples whose entries select from `1:nobs`.
  - `joint::Bool=false`: whether to score each split's test set as one predictive event with
    [`PosteriorStats.refit_joint_loglikelihoods`](@ref), rather than each held-out
    observation on its own, the estimand of [`loo`](@ref).
  - `normalize_by::Union{Symbol,Nothing}=joint ? nothing : :observation`: how scores are
    weighted when test sets overlap: `:observation` (each observation contributes its mean
    score; `joint=false` only), `:split` (each split contributes ``N/K``
    observation-equivalents) or `nothing` (the sum of all scores).
  - `save_fits::Bool=false`: whether to store the fit of each split in `fits`.
  - `ntasks::Int=1`: the maximum number of splits refit concurrently, as tasks that Julia
    schedules on its threads. With `ntasks > 1`, the wrapper must be safe to call from
    several tasks at once.

# References

- [Vehtari2017](@cite) Vehtari et al. Stat. Comput. 27 (2017).
- [LOOFAQ](@cite) Vehtari. Cross-validation FAQ.
"""
function cross_validate(
    rng::Random.AbstractRNG,
    wrapper,
    log_likelihood=nothing;
    folds=10,
    joint::Bool=false,
    normalize_by::Union{Nothing,Symbol}=joint ? nothing : :observation,
    save_fits::Bool=false,
    ntasks::Int=1,
)
    _check_normalize_by(normalize_by, joint)
    _check_ntasks(ntasks)
    nobservations = Int(StatsAPI.nobs(wrapper))
    splits = _splits(rng, folds, nobservations)
    lpd = _lpd_from_array(wrapper, log_likelihood, joint)
    lpd === nothing || _check_complementary_splits(splits, nobservations)
    refit_elpd = joint ? _refit_joint_elpd : _refit_pointwise_elpd
    results = _map_seeded(rng, splits, ntasks) do rng_split, split
        fit, estimates = refit_elpd(
            rng_split, wrapper, split.train_indices, split.test_indices
        )
        return (save_fits ? fit : nothing, estimates)
    end
    fits = save_fits ? map(first, results) : nothing
    estimates = map(last, results)
    pointwise = if joint
        _cv_joint(estimates, splits, normalize_by, nobservations)
    else
        _cv_pointwise(estimates, lpd, splits, normalize_by, nobservations)
    end
    return CrossValidationELPDResult(
        _elpd_estimates_from_pointwise(pointwise),
        pointwise,
        splits,
        fits,
        joint ? :split : :observation,
        normalize_by,
    )
end
function cross_validate(wrapper, log_likelihood=nothing; kwargs...)
    return cross_validate(Random.default_rng(), wrapper, log_likelihood; kwargs...)
end

function _check_normalize_by(normalize_by, joint::Bool)
    normalize_by ∈ (:observation, :split, nothing) || throw(
        ArgumentError(
            "`normalize_by` must be `:observation`, `:split` or `nothing`, got" *
            " `$(repr(normalize_by))`.",
        ),
    )
    if joint && normalize_by === :observation
        throw(
            ArgumentError(
                "`normalize_by=:observation` is not available with `joint=true`, since" *
                " each split is scored as one predictive event. Use `:split` or `nothing`.",
            ),
        )
    end
    return nothing
end

# `p` compares the scores with the in-sample log predictive density under the full-data
# posterior, which is the posterior given the training and held-out observations of a split
# only when the training set is the complement of the test set
function _check_complementary_splits(splits, nobservations)
    for (k, split) in enumerate(splits)
        ntest, ntrain = length(split.test_indices), length(split.train_indices)
        ntest + ntrain == nobservations && continue
        throw(
            ArgumentError(
                "`log_likelihood` can be given only when every split trains on all the" *
                " observations it does not hold out, but split $k holds out $ntest" *
                " observations and trains on $ntrain of $nobservations.",
            ),
        )
    end
    return nothing
end

# the in-sample log predictive density of each observation from `log_likelihood`, in the
# shape of the observations, or `nothing` if it was not given
_lpd_from_array(wrapper, ::Nothing, joint::Bool) = nothing
function _lpd_from_array(wrapper, log_likelihood, joint::Bool)
    if joint
        throw(
            ArgumentError(
                "`log_likelihood` cannot be given with `joint=true`, since the in-sample" *
                " log-likelihood of a set of observations scored jointly is not" *
                " determined by their pointwise log-likelihoods.",
            ),
        )
    end
    if ndims(log_likelihood) < 3
        throw(
            ArgumentError(
                "`log_likelihood` must have shape `(draws, chains, observations...)`," *
                " but it has $(ndims(log_likelihood)) dimensions.",
            ),
        )
    end
    _check_nobs(wrapper, prod(size(log_likelihood)[3:end]), "log_likelihood")
    return _lpd_pointwise(log_likelihood, (1, 2))
end

# the number of splits holding out each observation
function _coverage(splits, nobservations)
    coverage = zeros(Int, nobservations)
    for split in splits, i in split.test_indices
        coverage[i] += 1
    end
    return coverage
end

# the weight of each split's scores, with `normalize_by` as documented in `cross_validate`;
# `:observation` weights vary with the observation, so the weight is a function of its index
function _score_weights(normalize_by, splits, nobservations, T)
    nsplits = length(splits)
    if normalize_by === :observation
        coverage = _coverage(splits, nobservations)
        return map(_ -> (i -> inv(T(coverage[i]))), splits)
    elseif normalize_by === :split
        return map(splits) do split
            w = T(nobservations) / (nsplits * length(split.test_indices))
            return Returns(w)
        end
    else
        return map(_ -> Returns(one(T)), splits)
    end
end

# combine the scores of each held-out observation on its own, the estimand `loo` also
# targets, weighted over the splits that held it out
function _cv_pointwise(estimates, lpd, splits, normalize_by, nobservations)
    T = mapreduce(x -> eltype(x.elpd), promote_type, estimates)
    score_weights = _score_weights(normalize_by, splits, nobservations, T)
    # the position of each observation in the pointwise arrays
    elpd, var_elpd, reff, weight = ntuple(_ -> _zeros_pointwise(lpd, nobservations, T), 4)
    cells = lpd === nothing ? eachindex(elpd) : _observation_cells(lpd)
    for (split, weight_of, estimates_split) in zip(splits, score_weights, estimates)
        for (j, i) in enumerate(split.test_indices)
            cell = cells[i]
            w = weight_of(i)
            elpd[cell] += w * estimates_split.elpd[j]
            # the scores come from different fits, so their Monte Carlo errors are
            # independent
            var_elpd[cell] += w^2 * estimates_split.se_elpd[j]^2
            reff[cell] += w * estimates_split.reff[j]
            weight[cell] += w
        end
    end
    se_elpd = sqrt.(var_elpd)
    reff ./= weight
    # the observations held out in no split have no estimates
    p = lpd === nothing ? similar(elpd, Missing) : lpd .* weight .- elpd
    unscored = map(iszero, weight)
    any(unscored) || return (; elpd, se_elpd, p, reff)
    return map((; elpd, se_elpd, p, reff)) do x
        x isa AbstractArray{Missing} && return x
        y = _allowmissing(x)
        for cell in eachindex(y, unscored)
            unscored[cell] && (y[cell] = missing)
        end
        return y
    end
end

# zeros of `T` in the shape of `lpd`, or a vector if there is no `lpd`
_zeros_pointwise(lpd, nobservations, T) = zeros(T, nobservations)
_zeros_pointwise(lpd::AbstractArray, nobservations, T) = fill!(similar(lpd, T), zero(T))

# weight the score of each split as a single predictive event
function _cv_joint(estimates, splits, normalize_by, nobservations)
    T = mapreduce(x -> typeof(x.elpd), promote_type, estimates)
    score_weights = _score_weights(normalize_by, splits, nobservations, T)
    w = map(
        (weight_of, split) -> weight_of(first(split.test_indices)), score_weights, splits
    )
    elpd = w .* map(x -> x.elpd, estimates)
    return (;
        elpd,
        se_elpd=w .* map(x -> x.se_elpd, estimates),
        p=similar(elpd, Missing),
        reff=map(x -> x.reff, estimates),
    )
end

# `folds` as a number of folds assigned uniformly at random
function _splits(rng::Random.AbstractRNG, k::Integer, nobservations::Int)
    return _splits_from_labels(kfold_split_random(rng, Int(k), nobservations))
end

# `folds` as `(train, test)` tuples if every entry is one, and otherwise as one label per
# observation, in the shape of the observations or in their column-major order
function _splits(::Random.AbstractRNG, folds::AbstractArray, nobservations::Int)
    _istuples(folds) && return _splits_from_tuples(folds, nobservations)
    return _splits_from_labels(folds, nobservations)
end

function _istuples(folds)
    folds isa AbstractVector || return false
    eltype(folds) <: Tuple && return true
    return !isempty(folds) && all(x -> x isa Tuple, folds)
end

function _splits_from_labels(labels::AbstractArray, nobservations::Int)
    if length(labels) != nobservations
        throw(
            DimensionMismatch(
                "`folds` has $(length(labels)) labels but there are $nobservations" *
                " observations.",
            ),
        )
    end
    return _splits_from_labels(vec(labels))
end

function _splits_from_labels(labels::AbstractVector)
    unique_labels = unique(labels)
    _check_nfolds(length(unique_labels), length(labels))
    return map(unique_labels) do label
        positions = findall(isequal(label), labels)
        return (;
            train_indices=setdiff(1:length(labels), positions), test_indices=positions
        )
    end
end

# `folds` as `(train, test)` tuples of selectors into `1:nobs`, used as given
function _splits_from_tuples(tuples::AbstractVector, nobservations::Int)
    isempty(tuples) && throw(ArgumentError("`folds` must contain at least one split."))
    indices = 1:nobservations
    return map(enumerate(tuples)) do (i, tuple)
        length(tuple) == 2 || throw(
            ArgumentError(
                "each entry of `folds` must be a `(train_indices, test_indices)` tuple," *
                " but a $(length(tuple))-tuple was given.",
            ),
        )
        split = (;
            train_indices=_select(indices, tuple[1], i),
            test_indices=_select(indices, tuple[2], i),
        )
        _check_split(i, split)
        return split
    end
end

# the observations selected by `selector`, which may be a collection of indices or any other
# index into `indices` such as `InvertedIndices.Not`, as a vector
function _select(indices, selector, i)
    selected = try
        indices[selector]
    catch e
        e isa BoundsError || rethrow()
        throw(
            ArgumentError(
                "split $i of `folds` selects observations outside the observation indices" *
                " $indices.",
            ),
        )
    end
    return vec(collect(selected))
end

function _check_split(i, split)
    for (name, split_indices) in pairs(split)
        isempty(split_indices) &&
            throw(ArgumentError("split $i of `folds` has empty $name."))
        allunique(split_indices) ||
            throw(ArgumentError("split $i of `folds` has repeated $name."))
    end
    overlap = intersect(split.train_indices, split.test_indices)
    isempty(overlap) || throw(
        ArgumentError(
            "split $i of `folds` has observations $overlap in both its training and its" *
            " test set, so they would be scored by a fit that has seen them.",
        ),
    )
    return nothing
end

function _check_nfolds(k::Int, n::Int)
    k > 1 || throw(ArgumentError("number of folds must be greater than 1, got $k."))
    k ≤ n || throw(
        ArgumentError(
            "number of folds ($k) must not exceed the number of observations ($n)."
        ),
    )
    return nothing
end

"""
    kfold_split_random([rng::Random.AbstractRNG,] k::Int, n::Int) -> Vector{Int}

Assign `n` observations to `k` folds of as equal size as possible, uniformly at random.

See also: [`cross_validate`](@ref), [`kfold_split_stratified`](@ref),
[`kfold_split_grouped`](@ref)

# Examples

```jldoctest
julia> using Random

julia> fold_ids = kfold_split_random(Xoshiro(4), 3, 7);

julia> [count(==(f), fold_ids) for f in 1:3]
3-element Vector{Int64}:
 3
 2
 2
```
"""
function kfold_split_random(rng::Random.AbstractRNG, k::Int, n::Int)
    _check_nfolds(k, n)
    fold_ids = mod1.(1:n, k)
    return Random.shuffle!(rng, fold_ids)
end
kfold_split_random(k::Int, n::Int) = kfold_split_random(Random.default_rng(), k, n)

"""
    kfold_split_stratified([rng::Random.AbstractRNG,] k::Int, strata) -> Vector{Int}

Assign observations to `k` folds, preserving the proportions of `strata` in every fold.

The result has the shape of `strata`. This is useful when a categorical outcome or covariate
is unbalanced, so that random folds would vary appreciably in composition.

See also: [`cross_validate`](@ref), [`kfold_split_random`](@ref),
[`kfold_split_grouped`](@ref)

# Examples

Each fold gets one of the two observations from each stratum:

```jldoctest
julia> using Random

julia> strata = [:a, :a, :b, :b, :c, :c];

julia> fold_ids = kfold_split_stratified(Xoshiro(8), 2, strata);

julia> map(f -> sort(strata[fold_ids .== f]), 1:2)
2-element Vector{Vector{Symbol}}:
 [:a, :b, :c]
 [:a, :b, :c]
```
"""
function kfold_split_stratified(rng::Random.AbstractRNG, k::Int, strata)
    strata_array = collect(strata)
    strata_vec = vec(strata_array)
    n = length(strata_vec)
    _check_nfolds(k, n)
    # shuffle within each stratum, then deal the shuffled observations round-robin
    order = Vector{Int}(undef, 0)
    for stratum in unique(strata_vec)
        positions = findall(isequal(stratum), strata_vec)
        append!(order, length(positions) > 1 ? Random.shuffle(rng, positions) : positions)
    end
    fold_ids = Vector{Int}(undef, n)
    for (j, position) in enumerate(order)
        fold_ids[position] = mod1(j, k)
    end
    return reshape(fold_ids, size(strata_array))
end
function kfold_split_stratified(k::Int, strata)
    return kfold_split_stratified(Random.default_rng(), k, strata)
end

"""
    kfold_split_grouped([rng::Random.AbstractRNG,] k::Int, groups) -> Vector{Int}

Assign observations to `k` folds, keeping the observations of each of `groups` together.

The result has the shape of `groups`, and there must be at least `k` distinct groups. This
is useful for clustered or repeated-measures data, where leaving out part of a group would
leak information from the group into the training set.

See also: [`cross_validate`](@ref), [`kfold_split_random`](@ref),
[`kfold_split_stratified`](@ref)

# Examples

```jldoctest
julia> using Random

julia> groups = [:a, :a, :a, :b, :b, :c, :c, :c];

julia> kfold_split_grouped(Xoshiro(5), 3, groups)
8-element Vector{Int64}:
 1
 1
 1
 2
 2
 3
 3
 3
```
"""
function kfold_split_grouped(rng::Random.AbstractRNG, k::Int, groups)
    levels = unique(groups)
    nlevels = length(levels)
    if nlevels < k
        throw(
            ArgumentError(
                "number of folds ($k) must not exceed $nlevels, the number of distinct" *
                " groups.",
            ),
        )
    end
    fold_of_level = nlevels == k ? collect(1:nlevels) : kfold_split_random(rng, k, nlevels)
    level_positions = Dict(level => i for (i, level) in enumerate(levels))
    return map(group -> fold_of_level[level_positions[group]], collect(groups))
end
function kfold_split_grouped(k::Int, groups)
    return kfold_split_grouped(Random.default_rng(), k, groups)
end
