using DimensionalData: DimArray, Dim, dims, name
using InvertedIndices: Not
using Distributions
using Logging: SimpleLogger, with_logger
using OffsetArrays
using PosteriorStats
using Random
using StatsAPI: StatsAPI, nobs
using Statistics: mean
using Tables: Tables
using Test

# `wrapper` with a count of its refits
function counting_wrapper(wrapper::CallableWrapper)
    nrefits = Ref(0)
    counted = CallableWrapper(
        nobs(wrapper);
        refit=(rng, train_indices, eval_indices) ->
            (nrefits[] += 1; wrapper.refit(rng, train_indices, eval_indices)),
        loglikelihoods=wrapper.loglikelihoods,
        joint_loglikelihoods=wrapper.joint_loglikelihoods,
    )
    return counted, nrefits
end

@testset "cross_validate" begin
    y = [0.3, -1.2, 0.7, 2.4, -0.1, 8.0]
    wrapper = conjugate_normal_wrapper(y)
    log_likelihood = PosteriorStats.refit_loglikelihoods(
        Xoshiro(0),
        wrapper,
        PosteriorStats.refit(Xoshiro(0), wrapper, eachindex(y), eachindex(y)),
        eachindex(y),
    )
    lpd = PosteriorStats._lpd_pointwise(log_likelihood, (1, 2))

    @testset "explicit folds recover the exact ELPD" begin
        fold_ids = [1, 1, 2, 2, 3, 3]
        result = cross_validate(Xoshiro(0), wrapper; folds=fold_ids)
        @test result isa CrossValidationELPDResult
        @test [split.test_indices for split in result.splits] == [[1, 2], [3, 4], [5, 6]]
        @test [split.train_indices for split in result.splits] == [[3, 4, 5, 6], [1, 2, 5, 6], [1, 2, 3, 4]]
        @test result.fits === nothing

        pointwise = elpd_estimates(result; pointwise=true)
        @test keys(pointwise) == (:elpd, :se_elpd, :p, :reff)
        @test pointwise.elpd isa Vector{Float64}
        @test pointwise.elpd ≈ conjugate_normal_exact_kfold_elpd(y, fold_ids) rtol = 0.05
        @test all(>(0), pointwise.se_elpd)
        @test all(isfinite, pointwise.reff)
        @test elpd_estimates(result).elpd ≈ sum(pointwise.elpd)
        @test isfinite(elpd_estimates(result).se_elpd)
        # the effective number of parameters needs the in-sample log-likelihood
        @test pointwise.p isa Vector{Missing}
        @test length(pointwise.p) == length(y)
        @test elpd_estimates(result).p === missing
        @test elpd_estimates(result).se_p === missing
    end

    @testset "the effective number of parameters from the log-likelihood array" begin
        fold_ids = [1, 1, 2, 2, 3, 3]
        counted, nrefits = counting_wrapper(wrapper)
        result = cross_validate(Xoshiro(0), counted, log_likelihood; folds=fold_ids)
        @test nrefits[] == 3
        without = cross_validate(Xoshiro(0), wrapper; folds=fold_ids)
        @test result.splits == without.splits
        pointwise = elpd_estimates(result; pointwise=true)
        @test pointwise.elpd == elpd_estimates(without; pointwise=true).elpd
        @test pointwise.se_elpd == elpd_estimates(without; pointwise=true).se_elpd
        # the lpd comes from the full-data posterior
        @test pointwise.elpd .+ pointwise.p ≈ lpd
        @test elpd_estimates(result).elpd == elpd_estimates(without).elpd
        @test elpd_estimates(result).p ≈ sum(pointwise.p)
        @test isfinite(elpd_estimates(result).se_p)

        @testset "observations not held out have no estimate" begin
            partial = cross_validate(
                Xoshiro(0), wrapper, log_likelihood; folds=[(Not(i), i) for i in 3:6]
            )
            pointwise_partial = elpd_estimates(partial; pointwise=true)
            @test all(ismissing, pointwise_partial.p[1:2])
            @test collect(pointwise_partial.elpd[3:6] .+ pointwise_partial.p[3:6]) ≈
                lpd[3:6]
            @test elpd_estimates(partial).p ≈ sum(pointwise_partial.p[3:6])
        end

        @testset "errors" begin
            # the full-data lpd matches a split only if its training set is the complement
            # of its test set
            @testset for folds in (
                [(1:(i - 1), i) for i in 2:6],          # trains on a prefix
                [(1:(i - 1), i:(i + 1)) for i in 2:5],  # sliding windows
                [(Not([i, i + 1]), i) for i in 1:5],    # leaves an observation out of both
            )
                @test_throws ArgumentError cross_validate(
                    Xoshiro(0), wrapper, log_likelihood; folds
                )
            end
            # the joint in-sample log-likelihood cannot be derived from the pointwise array
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper, log_likelihood; folds=fold_ids, joint=true
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper, log_likelihood[:, :, 1]; folds=3
            )
            @test_throws DimensionMismatch cross_validate(
                Xoshiro(0),
                conjugate_normal_wrapper(y[1:(end - 1)]),
                log_likelihood;
                folds=3,
            )
        end
    end

    @testset "closed-form predictive densities as a single draw" begin
        exact = conjugate_normal_exact_wrapper(y)
        fold_ids = [1, 1, 2, 2, 3, 3]
        result = cross_validate(Xoshiro(0), exact; folds=fold_ids)
        pointwise = elpd_estimates(result; pointwise=true)
        @test pointwise.elpd ≈ conjugate_normal_exact_kfold_elpd(y, fold_ids)
        # no Monte Carlo error
        @test all(iszero, pointwise.se_elpd)
        @test all(isone, pointwise.reff)
        @test isfinite(elpd_estimates(result).se_elpd)

        splits = [(setdiff(eachindex(y), t), t) for t in ([1, 2], [3, 4], [5, 6])]
        joint = cross_validate(Xoshiro(0), exact; folds=fold_ids, joint=true)
        @test elpd_estimates(joint; pointwise=true).elpd ≈
            conjugate_normal_exact_joint_elpd(y, splits)
        @test all(iszero, elpd_estimates(joint; pointwise=true).se_elpd)

        loo_result = with_logger(SimpleLogger(IOBuffer())) do
            loo(log_likelihood)
        end
        refit_result = reloo(Xoshiro(0), exact, loo_result; k_threshold=(-Inf))
        @test elpd_estimates(refit_result; pointwise=true).elpd ≈
            conjugate_normal_exact_loo_elpd(y)

        # the in-sample predictive densities may be given as a single draw too
        lpd_exact = PosteriorStats.refit_loglikelihoods(
            Xoshiro(0),
            exact,
            PosteriorStats.refit(Xoshiro(0), exact, eachindex(y), eachindex(y)),
            eachindex(y),
        )
        with_p = cross_validate(Xoshiro(0), exact, lpd_exact; folds=fold_ids)
        pointwise_p = elpd_estimates(with_p; pointwise=true)
        @test pointwise_p.p ≈ vec(lpd_exact) .- pointwise_p.elpd
        @test isfinite(elpd_estimates(with_p).se_p)
    end

    @testset "as many folds as observations is exact LOO" begin
        result = cross_validate(Xoshiro(0), wrapper; folds=length(y))
        @test length(result.splits) == length(y)
        pointwise = elpd_estimates(result; pointwise=true)
        @test pointwise.elpd ≈ conjugate_normal_exact_loo_elpd(y) rtol = 0.05
        @test occursin(
            "computed from leave-one-out cross-validation of 6 observations",
            sprint(show, "text/plain", result),
        )
        # but not when the training sets are not the complements
        prefixes = cross_validate(Xoshiro(0), wrapper; folds=[(1:(i - 1), i) for i in 2:6])
        @test occursin("5 splits covering 5 of 6", sprint(show, "text/plain", prefixes))
    end

    @testset "fold labels may be of any type" begin
        labels = [:a, :a, :b, :b, :c, :c]
        result = cross_validate(Xoshiro(0), wrapper; folds=labels)
        @test [split.test_indices for split in result.splits] == [[1, 2], [3, 4], [5, 6]]
        integer_result = cross_validate(Xoshiro(0), wrapper; folds=[1, 1, 2, 2, 3, 3])
        @test elpd_estimates(result; pointwise=true).elpd ==
            elpd_estimates(integer_result; pointwise=true).elpd
    end

    @testset "random folds are reproducible given an rng" begin
        result1 = cross_validate(Xoshiro(7), wrapper; folds=3)
        result2 = cross_validate(Xoshiro(7), wrapper; folds=3)
        @test result1.splits == result2.splits
        @test cross_validate(Xoshiro(7), wrapper; folds=Int32(3)).splits == result1.splits
        @test isequal(elpd_estimates(result1), elpd_estimates(result2))
        @test length(result1.splits) == 3
        @test sort(map(split -> length(split.test_indices), result1.splits)) == [2, 2, 2]
        @test cross_validate(Xoshiro(7), wrapper, log_likelihood; folds=3).splits ==
            result1.splits
    end

    @testset "save_fits" begin
        fold_ids = [1, 1, 2, 2, 3, 3]
        result = cross_validate(Xoshiro(0), wrapper; folds=fold_ids, save_fits=true)
        @test result.fits isa Vector{Matrix{Float64}}
        @test length(result.fits) == 3
        # each stored fit is what the wrapper returned for that split's training set
        for (split, fit) in zip(result.splits, result.fits)
            posterior = conjugate_normal_posterior(y, split.train_indices)
            @test size(fit) == (1_000, 4)
            @test mean(fit) ≈ posterior.μ atol = 5 * posterior.σ / sqrt(length(fit))
        end
        # the exact wrapper's fits are its predictive and posterior
        exact = conjugate_normal_exact_wrapper(y)
        result_exact = cross_validate(Xoshiro(0), exact; folds=fold_ids, save_fits=true)
        for (split, fit) in zip(result_exact.splits, result_exact.fits)
            @test fit == PosteriorStats.refit(
                Xoshiro(0), exact, split.train_indices, split.test_indices
            )
        end
    end

    @testset "the refits are seeded from the rng" begin
        fold_ids = [1, 1, 2, 2, 3, 3]
        @testset for rng in (Xoshiro(3), MersenneTwister(3))
            result = cross_validate(copy(rng), wrapper; folds=fold_ids, save_fits=true)
            @test isequal(
                cross_validate(copy(rng), wrapper; folds=fold_ids, save_fits=true).fits,
                result.fits,
            )
            @test !isequal(
                cross_validate(wrapper; folds=fold_ids, save_fits=true).fits, result.fits
            )
            # the rng passed to the wrapper is a seeded copy of the rng given
            recorded = CallableWrapper(
                length(y);
                refit=(rng_split, train_indices, eval_indices) -> begin
                    @test typeof(rng_split) === typeof(copy(rng))
                    @test rng_split !== rng
                    wrapper.refit(rng_split, train_indices, eval_indices)
                end,
                loglikelihoods=wrapper.loglikelihoods,
            )
            cross_validate(copy(rng), recorded; folds=fold_ids)
            # neither the splits nor the fits depend on `ntasks`
            @testset for ntasks in (2, 5)
                parallel = cross_validate(
                    copy(rng), wrapper; folds=fold_ids, save_fits=true, ntasks
                )
                @test parallel.splits == result.splits
                @test isequal(parallel.fits, result.fits)
                @test isequal(
                    elpd_estimates(parallel; pointwise=true),
                    elpd_estimates(result; pointwise=true),
                )
                random = cross_validate(copy(rng), wrapper; folds=3, ntasks)
                @test random.splits == cross_validate(copy(rng), wrapper; folds=3).splits
                @test isequal(
                    elpd_estimates(random; pointwise=true),
                    elpd_estimates(
                        cross_validate(copy(rng), wrapper; folds=3); pointwise=true
                    ),
                )
            end
        end
        @test_throws ArgumentError cross_validate(
            Xoshiro(0), wrapper; folds=fold_ids, ntasks=0
        )
        # the fits run concurrently, and a failure in one stops the others
        failing = CallableWrapper(
            length(y);
            refit=(_, train_indices, _) ->
                length(train_indices) == 4 ? error("boom") : nothing,
            loglikelihoods=(_, _, eval_indices) -> zeros(1, 1, length(eval_indices)),
        )
        @test_throws TaskFailedException cross_validate(
            Xoshiro(0), failing; folds=fold_ids, ntasks=2
        )
        @test_throws ErrorException cross_validate(Xoshiro(0), failing; folds=fold_ids)
    end

    @testset "AbstractELPDResult interface" begin
        result = cross_validate(Xoshiro(0), wrapper; folds=3)
        estimates = elpd_estimates(result)
        @test information_criterion(result, :log) == estimates.elpd
        @test information_criterion(result, :negative_log) == -estimates.elpd
        @test information_criterion(result, :deviance) == -2 * estimates.elpd
    end

    @testset "compare accepts the result" begin
        loo_result = with_logger(SimpleLogger(IOBuffer())) do
            loo(log_likelihood)
        end
        @testset for with_p in (false, true)
            result = if with_p
                cross_validate(Xoshiro(0), wrapper, log_likelihood; folds=3)
            else
                cross_validate(Xoshiro(0), wrapper; folds=3)
            end
            comparison = compare((psis=loo_result, cv=result))
            @test comparison isa ModelComparisonResult
            @test Set(comparison.name) == Set((:psis, :cv))
            @test all(isfinite, comparison.se_elpd_diff)
            # the table reports p where it is defined
            table = Tables.columntable(comparison)
            p = Dict(zip(table.name, table.p))
            @test isfinite(p[:psis])
            @test with_p ? isfinite(p[:cv]) : ismissing(p[:cv])
            @test occursin(r"\bp\s+se_p", sprint(show, "text/plain", comparison))
        end
    end

    @testset "eltype is preserved" begin
        y32 = Float32.(y)
        wrapper32 = conjugate_normal_wrapper(y32; ndraws=100)
        log_likelihood32 = PosteriorStats.refit_loglikelihoods(
            Xoshiro(0),
            wrapper32,
            PosteriorStats.refit(Xoshiro(0), wrapper32, eachindex(y32), eachindex(y32)),
            eachindex(y32),
        )
        @testset for args in ((), (log_likelihood32,))
            result = cross_validate(Xoshiro(0), wrapper32, args...; folds=3)
            pointwise = elpd_estimates(result; pointwise=true)
            @test eltype(pointwise.elpd) === Float32
            @test eltype(pointwise.se_elpd) === Float32
            @test elpd_estimates(result).elpd isa Float32
            if !isempty(args)
                @test eltype(pointwise.p) === Float32
                @test elpd_estimates(result).p isa Float32
            end
        end
    end

    @testset "pointwise estimates keep the axes of the log-likelihood array" begin
        # the observations are numbered 1:nobs whatever the axes of the array
        offset = 10
        log_likelihoodo = OffsetArray(log_likelihood, 0, 0, offset)
        fold_ids = [1, 1, 2, 2, 3, 3]
        result = cross_validate(Xoshiro(0), wrapper, log_likelihoodo; folds=fold_ids)
        pointwise = elpd_estimates(result; pointwise=true)
        @test axes(pointwise.elpd) == (axes(log_likelihoodo, 3),)
        @test axes(pointwise.p) == (axes(log_likelihoodo, 3),)
        @test [split.test_indices for split in result.splits] == [[1, 2], [3, 4], [5, 6]]
        @test collect(pointwise.elpd) ≈ conjugate_normal_exact_kfold_elpd(y, fold_ids) rtol =
            0.05
    end

    @testset "errors" begin
        @test_throws ArgumentError cross_validate(Xoshiro(0), wrapper; folds=1)
        @test_throws ArgumentError cross_validate(Xoshiro(0), wrapper; folds=length(y) + 1)
        @test_throws ArgumentError cross_validate(
            Xoshiro(0), wrapper; folds=fill(1, length(y))
        )
        @test_throws DimensionMismatch cross_validate(
            Xoshiro(0), wrapper; folds=[1, 1, 2, 2]
        )

        bad_wrapper = CallableWrapper(
            length(y); refit=(_, _, _) -> nothing, loglikelihoods=(_, _, _) -> randn(100, 4)
        )
        @test_throws DimensionMismatch cross_validate(Xoshiro(0), bad_wrapper; folds=3)
    end

    @testset "(train, test) pairs" begin
        @testset "pairs and labels agree when they describe the same partition" begin
            pairs = [([3, 4, 5, 6], [1, 2]), ([1, 2, 5, 6], [3, 4]), ([1, 2, 3, 4], [5, 6])]
            from_pairs = cross_validate(Xoshiro(0), wrapper; folds=pairs)
            from_labels = cross_validate(Xoshiro(0), wrapper; folds=[1, 1, 2, 2, 3, 3])
            @test from_pairs.splits == from_labels.splits
            @test isequal(elpd_estimates(from_pairs), elpd_estimates(from_labels))
            @test elpd_estimates(from_pairs; pointwise=true).elpd ==
                elpd_estimates(from_labels; pointwise=true).elpd
        end

        @testset "training sets need not be complements" begin
            # rolling-origin splits: train on a prefix, score the next block
            pairs = [(1:2, 3:4), (1:4, 5:6)]
            result = cross_validate(Xoshiro(0), wrapper; folds=pairs)
            @test [split.train_indices for split in result.splits] == [[1, 2], [1, 2, 3, 4]]
            pointwise = elpd_estimates(result; pointwise=true)
            # the first two observations are in no test set
            @test all(ismissing, pointwise.elpd[1:2])
            @test all(!ismissing, pointwise.elpd[3:6])
            @test pointwise.p isa Vector{Missing}
            # totals come from the tested observations only
            @test elpd_estimates(result).elpd ≈ sum(pointwise.elpd[3:6])
            @test isfinite(elpd_estimates(result).se_elpd)
            # the full-data lpd does not match a prefix fit, so p cannot be computed
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper, log_likelihood; folds=pairs
            )
            # and each block is scored against its own prefix fit
            expected = map(pairs) do (train_indices, test_indices)
                posterior = conjugate_normal_posterior(y, train_indices)
                predictive = Normal(posterior.μ, hypot(1, posterior.σ))
                return logpdf.(predictive, y[test_indices])
            end
            @test collect(skipmissing(pointwise.elpd)) ≈ vcat(expected...) rtol = 0.05

            @testset "comparison and weights use the jointly scored observations" begin
                other = cross_validate(Xoshiro(1), wrapper; folds=pairs)
                full = cross_validate(Xoshiro(0), wrapper; folds=[1, 1, 2, 2, 3, 3])
                @testset for method in (
                    Stacking(), PseudoBMA(), BootstrappedPseudoBMA(; rng=Xoshiro(0))
                )
                    comparison = compare((a=result, b=other); weights_method=method)
                    @test all(isfinite, comparison.elpd_diff)
                    @test all(isfinite, comparison.se_elpd_diff)
                    @test sum(comparison.weight) ≈ 1
                    # observations one result did not score are left out of the differences
                    comparison = compare((full=full, partial=result); weights_method=method)
                    @test all(isfinite, comparison.elpd_diff)
                    @test sum(comparison.weight) ≈ 1
                end
                elpd_full = elpd_estimates(full; pointwise=true).elpd
                comparison = compare((full=full, partial=result); sort=false)
                expected = abs(sum(elpd_full[3:6] .- pointwise.elpd[3:6]))
                @test maximum(abs, comparison.elpd_diff) ≈ expected
            end
        end

        @testset "pairs are recognized by their entries" begin
            pairs = [(1:2, 3:4), (1:4, 5:6)]
            expected = cross_validate(Xoshiro(0), wrapper; folds=pairs).splits
            untyped = Any[]
            foreach(pair -> push!(untyped, pair), pairs)
            @test cross_validate(Xoshiro(0), wrapper; folds=untyped).splits == expected
            # `Pair`s are not splits, since a split is not a key-value mapping
            @test_throws DimensionMismatch cross_validate(
                Xoshiro(0), wrapper; folds=[train => test for (train, test) in pairs]
            )
            # one pair per observation is not mistaken for one label per observation
            loo_pairs = Any[(Not(j), j) for j in 1:6]
            @test cross_validate(Xoshiro(0), wrapper; folds=loo_pairs).splits ==
                cross_validate(Xoshiro(0), wrapper; folds=1:6).splits
        end

        @testset "entries are selectors into 1:nobs" begin
            from_labels = cross_validate(Xoshiro(0), wrapper; folds=1:6)
            @testset for pairs in (
                [(Not(j), [j]) for j in 1:6],   # `Not(j)` selects the complement
                [(Not(j), j) for j in 1:6],     # a single index
                [(Not(j), j:j) for j in 1:6],   # a range
                [(map(!=(j), 1:6), map(==(j), 1:6)) for j in 1:6],   # a mask
            )
                @test cross_validate(Xoshiro(0), wrapper; folds=pairs).splits ==
                    from_labels.splits
            end
        end

        @testset "a single holdout split" begin
            result = cross_validate(Xoshiro(0), wrapper; folds=[(1:4, 5:6)])
            @test length(result.splits) == 1
            pointwise = elpd_estimates(result; pointwise=true)
            @test count(!ismissing, pointwise.elpd) == 2
        end

        @testset "errors" begin
            @test_throws ArgumentError cross_validate(Xoshiro(0), wrapper; folds=Tuple{}[])
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=[(1:2, 3:4, 5:6)]
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=[(1:4, 5:7)]
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=[(1:4, Int[])]
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=[(Int[], 1:4)]
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=[(1:4, [5, 5])]
            )
            # an observation in both the training and the test set of one split
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=[(1:4, 4:6)]
            )
        end
    end

    @testset "overlapping test sets" begin
        # sliding windows: forecast the next two observations from each origin
        pairs = [(1:t, (t + 1):(t + 2)) for t in 2:4]
        coverage = [0, 0, 1, 2, 2, 1]
        # the exact score of each evaluation, by observation
        scores = [Float64[] for _ in y]
        for (train_indices, test_indices) in pairs
            posterior = conjugate_normal_posterior(y, train_indices)
            predictive = Normal(posterior.μ, hypot(1, posterior.σ))
            for i in test_indices
                push!(scores[i], logpdf(predictive, y[i]))
            end
        end
        scored = coverage .> 0

        @testset "each observation's estimate is the mean of its scores" begin
            result = cross_validate(Xoshiro(0), wrapper; folds=pairs)
            @test result.normalize_by === :observation
            pointwise = elpd_estimates(result; pointwise=true)
            @test all(ismissing, pointwise.elpd[.!scored])
            @test collect(skipmissing(pointwise.elpd)) ≈ mean.(scores[scored]) rtol = 0.05
            @test all(>(0), skipmissing(pointwise.se_elpd))
            @test all(isfinite, skipmissing(pointwise.reff))
            @test elpd_estimates(result).elpd ≈ sum(skipmissing(pointwise.elpd))
            @test occursin("up to 2 times each", sprint(show, "text/plain", result))
        end

        @testset "or the sum, or normalized by split" begin
            summed = cross_validate(Xoshiro(0), wrapper; folds=pairs, normalize_by=nothing)
            @test collect(skipmissing(elpd_estimates(summed; pointwise=true).elpd)) ≈
                sum.(scores[scored]) rtol = 0.05
            by_split = cross_validate(Xoshiro(0), wrapper; folds=pairs, normalize_by=:split)
            # N / (K |E_k|) = 1 here, so the weights coincide with the sum
            @test isequal(
                elpd_estimates(by_split; pointwise=true).elpd,
                elpd_estimates(summed; pointwise=true).elpd,
            )
            @test occursin("summing the scores", sprint(show, "text/plain", summed))
            @test occursin("normalized by split", sprint(show, "text/plain", by_split))
        end

        @testset "the Monte Carlo errors of the scores are independent" begin
            result = cross_validate(Xoshiro(0), wrapper; folds=pairs)
            single = cross_validate(Xoshiro(0), wrapper; folds=pairs[2:2])   # observations 4 and 5, once each
            se = elpd_estimates(result; pointwise=true).se_elpd
            se_single = elpd_estimates(single; pointwise=true).se_elpd
            # the mean of two scores has less error than either
            @test se[4] < se_single[4]
            @test se[5] < se_single[5]
        end

        @testset "p weights the in-sample lpd like the scores" begin
            # leave-one-out splits that hold out some observations more than once
            loo_pairs = [(Not(i), i) for i in [2, 3, 3, 5, 5, 5]]
            loo_coverage = [0, 1, 2, 0, 3, 0]
            @testset for normalize_by in (:observation, :split, nothing)
                result = cross_validate(
                    Xoshiro(0), wrapper, log_likelihood; folds=loo_pairs, normalize_by
                )
                pointwise = elpd_estimates(result; pointwise=true)
                # the weights of each observation's scores sum to 1, to N / K times its
                # coverage, or to its coverage
                w = if normalize_by === :observation
                    ones(length(y))
                elseif normalize_by === :split
                    loo_coverage .* (length(y) / length(loo_pairs))
                else
                    loo_coverage
                end
                expected = lpd .* w .- pointwise.elpd
                @test collect(skipmissing(pointwise.p)) ≈ collect(skipmissing(expected))
                @test all(ismissing, pointwise.p[loo_coverage .== 0])
            end
        end

        @testset "the shape of the log-likelihood array is kept" begin
            log_likelihoodm = reshape(log_likelihood, 1_000, 4, 2, 3)
            wrapperm = conjugate_normal_wrapper(reshape(y, 2, 3))
            loo_pairs = [(Not(i), i) for i in 3:6]
            result = cross_validate(Xoshiro(0), wrapperm, log_likelihoodm; folds=loo_pairs)
            pointwise = elpd_estimates(result; pointwise=true)
            @test size(pointwise.elpd) == (2, 3)
            @test isequal(
                vec(pointwise.elpd),
                elpd_estimates(
                    cross_validate(Xoshiro(0), wrapper; folds=loo_pairs); pointwise=true
                ).elpd,
            )
        end
    end

    @testset "joint predictive units" begin
        fold_ids = [1, 1, 2, 2, 3, 3]
        splits = [(setdiff(eachindex(y), t), t) for t in ([1, 2], [3, 4], [5, 6])]
        result = cross_validate(Xoshiro(0), wrapper; folds=fold_ids, joint=true)
        pointwise = elpd_estimates(result; pointwise=true)

        @test result.estimand === :split
        @test length(pointwise.elpd) == length(result.splits) == 3
        # each split is scored as one predictive event
        @test pointwise.elpd ≈ conjugate_normal_exact_joint_elpd(y, splits) rtol = 0.02
        @test elpd_estimates(result).elpd ≈ sum(pointwise.elpd)
        @test all(>(0), pointwise.se_elpd)
        @test all(isfinite, pointwise.reff)
        @test pointwise.p isa Vector{Missing}
        @test length(pointwise.p) == 3
        @test elpd_estimates(result).p === missing

        @testset "the joint estimand differs from the pointwise one" begin
            pointwise_result = cross_validate(Xoshiro(0), wrapper; folds=fold_ids)
            @test elpd_estimates(result).elpd != elpd_estimates(pointwise_result).elpd
            @test PosteriorStats._elpd_estimand(pointwise_result) === :observation
        end

        @testset "test sets may overlap" begin
            # sliding windows of two observations
            pairs = [(1:t, (t + 1):(t + 2)) for t in 2:4]
            sliding = cross_validate(Xoshiro(0), wrapper; folds=pairs, joint=true)
            @test length(sliding.splits) == 3
            @test sliding.normalize_by === nothing
            @test elpd_estimates(sliding; pointwise=true).elpd ≈
                conjugate_normal_exact_joint_elpd(y, pairs) rtol = 0.02
            # results with different numbers of splits cannot be compared
            two = cross_validate(Xoshiro(0), wrapper; folds=pairs[1:2], joint=true)
            @test_throws ArgumentError compare((a=result, b=two))
        end

        @testset "weights" begin
            pairs = [(1:2, 3:4), (1:2, 3:6)]
            unweighted = cross_validate(Xoshiro(0), wrapper; folds=pairs, joint=true)
            weighted = cross_validate(
                Xoshiro(0), wrapper; folds=pairs, joint=true, normalize_by=:split
            )
            @test weighted.normalize_by === :split
            # each split contributes N / K observation-equivalents
            w = [6 / (2 * 2), 6 / (2 * 4)]
            @test elpd_estimates(weighted; pointwise=true).elpd ≈
                w .* elpd_estimates(unweighted; pointwise=true).elpd
            @test elpd_estimates(weighted; pointwise=true).se_elpd ≈
                w .* elpd_estimates(unweighted; pointwise=true).se_elpd
            @test elpd_estimates(weighted).elpd ≈
                sum(elpd_estimates(weighted; pointwise=true).elpd)
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=pairs, joint=true, normalize_by=:observation
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), wrapper; folds=pairs, normalize_by=:mean
            )
        end

        @testset "save_fits" begin
            saved = cross_validate(
                Xoshiro(0), wrapper; folds=fold_ids, joint=true, save_fits=true
            )
            @test length(saved.fits) == 3
            # the same fits as with pointwise scoring, since the seeds are the same
            pointwise_saved = cross_validate(
                Xoshiro(0), wrapper; folds=fold_ids, save_fits=true
            )
            @test saved.fits == pointwise_saved.fits
            @testset for ntasks in (2, 3)
                parallel = cross_validate(
                    Xoshiro(0), wrapper; folds=fold_ids, joint=true, save_fits=true, ntasks
                )
                @test parallel.fits == saved.fits
                @test isequal(elpd_estimates(parallel; pointwise=true), pointwise)
            end
        end

        @testset "AbstractELPDResult interface" begin
            @test information_criterion(result, :deviance) ==
                -2 * elpd_estimates(result).elpd
        end

        @testset "errors" begin
            # a wrapper with no `joint_loglikelihoods` callable
            bare = CallableWrapper(
                length(y);
                refit=(rng, train_indices, _) ->
                    rand(rng, conjugate_normal_posterior(y, train_indices), 100, 4),
                loglikelihoods=(_, mu, eval_indices) ->
                    logpdf.(Normal.(mu), reshape(y[eval_indices], 1, 1, :)),
            )
            @test_throws ArgumentError cross_validate(
                Xoshiro(0), bare; folds=fold_ids, joint=true
            )

            # a callable returning one value per observation instead of one per draw
            pointwise_by_mistake = CallableWrapper(
                length(y);
                refit=bare.refit,
                loglikelihoods=bare.loglikelihoods,
                joint_loglikelihoods=bare.loglikelihoods,
            )
            @test_throws DimensionMismatch cross_validate(
                Xoshiro(0), pointwise_by_mistake; folds=fold_ids, joint=true
            )
        end

        @testset "compare and model_weights refuse to mix predictive units" begin
            pointwise_result = cross_validate(Xoshiro(0), wrapper; folds=fold_ids)
            @test_throws ArgumentError compare((joint=result, pointwise=pointwise_result))
            @testset for method in
                         (Stacking(), PseudoBMA(), BootstrappedPseudoBMA(; rng=Xoshiro(0)))
                @test_throws ArgumentError model_weights(
                    (joint=result, pointwise=pointwise_result); method
                )
            end
        end

        @testset "show" begin
            @test sprint(show, "text/plain", result) == """
                CrossValidationELPDResult with estimates
                   elpd  se_elpd
                 -4e+01       23

                computed from 3 splits of 2 held-out observations each, scored jointly"""
            unequal = cross_validate(
                Xoshiro(0), wrapper; folds=[(1:2, 3:4), (1:2, 3:6)], joint=true
            )
            @test occursin(
                "computed from 2 splits of 2 to 4 held-out observations each, scored jointly",
                sprint(show, "text/plain", unequal),
            )
        end
    end

    @testset "observations with more than one dimension" begin
        # observations arranged as (time, group); every cell is an iid draw of the same model
        ym = reshape([y; 1.1; -0.4; 0.2; 0.9; -2.0; 0.5], 3, 4)
        wrapperm = conjugate_normal_wrapper(ym)
        # the observations are numbered in column-major order, so linear indexing selects
        # them from the matrix
        indices = 1:length(ym)
        log_likelihoodm = reshape(
            PosteriorStats.refit_loglikelihoods(
                Xoshiro(0),
                wrapperm,
                PosteriorStats.refit(Xoshiro(0), wrapperm, indices, indices),
                indices,
            ),
            1_000,
            4,
            size(ym)...,
        )
        @test size(log_likelihoodm) == (1_000, 4, 3, 4)
        labels = [g for t in axes(ym, 1), g in axes(ym, 2)]   # one fold per column

        @testset "leave-one-group-out from a label matrix" begin
            result = cross_validate(Xoshiro(0), wrapperm; folds=labels)
            @test length(result.splits) == 4
            # each split holds out one whole column
            @test result.splits[2].test_indices == vec(LinearIndices(ym)[:, 2]) == [4, 5, 6]
            @test PosteriorStats.slice_axes(
                CartesianIndices(ym)[result.splits[2].train_indices]
            ) == ([1, 2, 3], [1, 3, 4])
            pointwise = elpd_estimates(result; pointwise=true)
            # the pointwise estimates are a vector in the column-major order of the cells
            @test pointwise.elpd isa Vector
            @test reshape(pointwise.elpd, size(ym)) ≈
                conjugate_normal_exact_kfold_elpd(ym, labels) rtol = 0.05

            @testset "with the log-likelihood array they take the shape of the array" begin
                from_array = cross_validate(
                    Xoshiro(0), wrapperm, log_likelihoodm; folds=labels
                )
                @test from_array.splits == result.splits
                pointwise_array = elpd_estimates(from_array; pointwise=true)
                @test size(pointwise_array.elpd) == size(ym)
                @test vec(pointwise_array.elpd) == pointwise.elpd
                @test pointwise_array.elpd .+ pointwise_array.p ≈
                    PosteriorStats._lpd_pointwise(log_likelihoodm, (1, 2))
            end
        end

        @testset "pairs may select cells in several ways" begin
            expected = cross_validate(Xoshiro(0), wrapperm; folds=labels).splits
            blocks = LinearIndices(ym)
            @testset for pairs in (
                [(blocks[:, Not(j)], blocks[:, j]) for j in 1:4],           # blocks of cells
                [(Not(vec(blocks[:, j])), vec(blocks[:, j])) for j in 1:4], # complement of cells
            )
                result = cross_validate(Xoshiro(0), wrapperm; folds=pairs)
                @test result.splits == expected
                @test result.splits[1].train_indices isa Vector{Int}
            end
        end

        @testset "labels may also be given in column-major order" begin
            from_matrix = cross_validate(Xoshiro(0), wrapperm; folds=labels)
            from_vector = cross_validate(Xoshiro(0), wrapperm; folds=vec(labels))
            @test from_matrix.splits == from_vector.splits
        end

        @testset "random folds over cells" begin
            result = cross_validate(Xoshiro(1), wrapperm, log_likelihoodm; folds=4)
            @test length(result.splits) == 4
            @test eltype(result.splits[1].test_indices) === Int
            @test size(elpd_estimates(result; pointwise=true).elpd) == size(ym)
            @test elpd_estimates(result).elpd ≈
                sum(elpd_estimates(result; pointwise=true).elpd)
        end

        @testset "rolling origin over time, scored jointly" begin
            blocks = LinearIndices(ym)
            splits = [
                (vec(blocks[1:1, :]), vec(blocks[2:2, :])),
                (vec(blocks[1:2, :]), vec(blocks[3:3, :])),
            ]
            result = cross_validate(Xoshiro(0), wrapperm; folds=splits, joint=true)
            @test result.estimand === :split
            @test elpd_estimates(result; pointwise=true).elpd ≈
                conjugate_normal_exact_joint_elpd(ym, splits) rtol = 0.02
        end

        @testset "compare with loo of the array" begin
            loo_result = with_logger(SimpleLogger(IOBuffer())) do
                loo(log_likelihoodm)
            end
            # pointwise estimates of different shapes are compared in column-major order
            comparison = compare((
                psis=loo_result, cv=cross_validate(Xoshiro(0), wrapperm; folds=labels)
            ))
            @test all(isfinite, comparison.se_elpd_diff)
        end

        @testset "reloo holds out single cells" begin
            loo_result = with_logger(SimpleLogger(IOBuffer())) do
                loo(log_likelihoodm)
            end
            result = reloo(Xoshiro(0), wrapperm, loo_result; k_threshold=(-Inf))
            @test result.refit_indices == indices
            pointwise = elpd_estimates(result; pointwise=true)
            @test size(pointwise.elpd) == size(ym)
            @test pointwise.elpd ≈ conjugate_normal_exact_loo_elpd(ym) rtol = 0.05
            @test all(ismissing, pointwise.pareto_shape)
        end

        @testset "slice_axes" begin
            cells = CartesianIndices(ym)
            @test PosteriorStats.slice_axes(vec(cells[[1, 3], 2:4])) == ([1, 3], [2, 3, 4])
            # a set that is not a rectangle, such as all cells but one
            @test_throws ArgumentError PosteriorStats.slice_axes(vec(cells)[2:end])
            # duplicates cannot hide a missing corner
            @test_throws ArgumentError PosteriorStats.slice_axes([
                CartesianIndex(1, 1), CartesianIndex(1, 1), CartesianIndex(2, 2)
            ])
        end

        @testset "split helpers keep the shape of their input" begin
            strata = [g for t in 1:3, g in 1:4]
            @test size(kfold_split_stratified(Xoshiro(1), 2, strata)) == size(strata)
            @test size(kfold_split_grouped(Xoshiro(1), 4, strata)) == size(strata)
        end

        @testset "named observation dimensions are preserved" begin
            log_likelihood_dim = DimArray(
                log_likelihoodm,
                (Dim{:draw}(1:1_000), Dim{:chain}(1:4), Dim{:time}(1:3), Dim{:group}(1:4)),
            )
            result = cross_validate(Xoshiro(0), wrapperm, log_likelihood_dim; folds=labels)
            pointwise = elpd_estimates(result; pointwise=true)
            @test pointwise.elpd isa DimArray
            @test name.(dims(pointwise.elpd)) == (:time, :group)
            @test pointwise.elpd ≈ conjugate_normal_exact_kfold_elpd(ym, labels) rtol = 0.05

            loo_result = with_logger(SimpleLogger(IOBuffer())) do
                loo(log_likelihood_dim)
            end
            refit_result = reloo(Xoshiro(0), wrapperm, loo_result; k_threshold=(-Inf))
            @test elpd_estimates(refit_result; pointwise=true).elpd isa DimArray
            @test name.(dims(elpd_estimates(refit_result; pointwise=true).elpd)) ==
                (:time, :group)
        end
    end

    @testset "show" begin
        result = cross_validate(
            Xoshiro(0), wrapper, log_likelihood; folds=[1, 1, 2, 2, 3, 3]
        )
        @test sprint(show, "text/plain", result) == """
            CrossValidationELPDResult with estimates
               elpd  se_elpd  p  se_p
             -4e+01       23  7   5.4

            computed from 3-fold cross-validation of 6 observations"""

        # estimates the result does not compute are omitted
        without = cross_validate(Xoshiro(0), wrapper; folds=[1, 1, 2, 2, 3, 3])
        @test sprint(show, "text/plain", without) == """
            CrossValidationELPDResult with estimates
               elpd  se_elpd
             -4e+01       23

            computed from 3-fold cross-validation of 6 observations"""

        partial = cross_validate(Xoshiro(0), wrapper; folds=[(1:2, 3:4), (1:4, 5:6)])
        @test occursin(
            "computed from 2 splits covering 4 of 6 observations",
            sprint(show, "text/plain", partial),
        )
    end
end

@testset "fold splitting" begin
    @testset "kfold_split_random" begin
        @testset for (k, n) in ((2, 6), (3, 7), (5, 5), (3, 100))
            fold_ids = kfold_split_random(Xoshiro(1), k, n)
            @test length(fold_ids) == n
            @test sort(unique(fold_ids)) == 1:k
            sizes = [count(isequal(f), fold_ids) for f in 1:k]
            # folds are as equal in size as possible
            @test maximum(sizes) - minimum(sizes) ≤ 1
            @test sum(sizes) == n
        end

        @test kfold_split_random(Xoshiro(1), 3, 9) == kfold_split_random(Xoshiro(1), 3, 9)
        @test_throws ArgumentError kfold_split_random(1, 10)
        @test_throws ArgumentError kfold_split_random(11, 10)
    end

    @testset "kfold_split_stratified" begin
        strata = repeat([:a, :b], inner=6)
        fold_ids = kfold_split_stratified(Xoshiro(2), 3, strata)
        @test length(fold_ids) == length(strata)
        @test sort(unique(fold_ids)) == 1:3
        # every fold has the same stratum composition
        @testset for f in 1:3
            @test sort(strata[fold_ids .== f]) == [:a, :a, :b, :b]
        end

        @testset "unbalanced strata keep their proportions as closely as possible" begin
            strata = [fill(:a, 9); fill(:b, 3)]
            fold_ids = kfold_split_stratified(Xoshiro(2), 3, strata)
            @testset for f in 1:3
                @test count(isequal(:a), strata[fold_ids .== f]) == 3
                @test count(isequal(:b), strata[fold_ids .== f]) == 1
            end
        end

        @test_throws ArgumentError kfold_split_stratified(1, [1, 2, 3])
        @test_throws ArgumentError kfold_split_stratified(4, [1, 2, 3])
    end

    @testset "kfold_split_grouped" begin
        groups = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
        @testset "groups are never split across folds" begin
            @testset for k in (2, 3, 4)
                fold_ids = kfold_split_grouped(Xoshiro(3), k, groups)
                @test length(fold_ids) == length(groups)
                @test sort(unique(fold_ids)) == 1:k
                @testset for group in unique(groups)
                    @test length(unique(fold_ids[groups .== group])) == 1
                end
            end
        end

        @testset "one fold per group when k equals the number of groups" begin
            fold_ids = kfold_split_grouped(Xoshiro(3), 4, groups)
            @test fold_ids == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
        end

        @test_throws ArgumentError kfold_split_grouped(5, groups)
    end

    @testset "splits can be passed to cross_validate" begin
        y = randn(Xoshiro(11), 12)
        wrapper = conjugate_normal_wrapper(y; ndraws=200)
        strata = repeat([:a, :b], inner=6)
        @testset for fold_ids in (
            kfold_split_random(Xoshiro(1), 3, length(y)),
            kfold_split_stratified(Xoshiro(1), 3, strata),
            kfold_split_grouped(Xoshiro(1), 3, repeat(1:6, inner=2)),
        )
            result = cross_validate(Xoshiro(0), wrapper; folds=fold_ids)
            @test [split.test_indices for split in result.splits] == [findall(isequal(f), fold_ids) for f in unique(fold_ids)]
            @test elpd_estimates(result; pointwise=true).elpd ≈
                conjugate_normal_exact_kfold_elpd(y, fold_ids) rtol = 0.1
        end
    end
end
