using DimensionalData
using Distributions
using InferenceObjects
using PosteriorStats
using StatsBase
using Test

@testset "loo_pit" begin
    @testset "scalar data" begin
        ndraws = 100
        nchains = 3
        y = randn()
        y_pred = randn(ndraws, nchains)
        weights = rand(ndraws, nchains)
        log_weights = log.(weights) .- log(sum(weights))
        pitvals = @inferred loo_pit(y, y_pred, log_weights)
        @test pitvals isa typeof(y)
        @test 0 <= pitvals <= 1
        @test pitvals ≈ mean(y_pred .≤ y, StatsBase.weights(weights))
    end
    @testset "array data" begin
        ndraws = 100
        nchains = 3
        @testset for sz in ((100,), (5, 4)), T in (Float32, Float64)
            y = randn(T, sz...)
            y_pred = randn(T, ndraws, nchains, sz...)
            weights = rand(T, ndraws, nchains, sz...)
            weights ./= sum(weights; dims=(1, 2))
            log_weights = log.(weights)
            pitvals = @inferred loo_pit(y, y_pred, log_weights)
            @test pitvals isa typeof(y)
            @test size(pitvals) == sz
            @test all(p -> 0 ≤ p ≤ 1, pitvals)
            pitvals_exp = dropdims(
                sum((y_pred .≤ reshape(y, 1, 1, sz...)) .* weights; dims=(1, 2));
                dims=(1, 2),
            )
            @test pitvals ≈ pitvals_exp
        end
    end
    @testset "discrete data" begin
        ndraws = 1_000
        nchains = 3
        dists = Binomial.(10:10:100, 0.25)
        d = product_distribution(dists)
        y = rand(d)
        y_sample = rand(d, ndraws * nchains)
        y_pred = reshape(transpose(y_sample), ndraws, nchains, length(y))
        loglike = mapslices(yi -> logpdf.(dists, yi), y_pred; dims=3)
        log_weights = psis(loglike).log_weights
        pit_vals = loo_pit(
            PosteriorStats.smooth_data(y; dims=1),
            PosteriorStats.smooth_data(y_pred; dims=3),
            log_weights,
        )
        ϵ = sqrt(eps())
        @test loo_pit(y, y_pred, log_weights) == pit_vals
        @test loo_pit(y, y_pred, log_weights; is_discrete=true) == pit_vals
        @test loo_pit(y, y_pred, log_weights; is_discrete=false) != pit_vals
        @test !(loo_pit(y .+ ϵ, y_pred, log_weights) ≈ pit_vals)
        @test loo_pit(y .+ ϵ, y_pred, log_weights; is_discrete=true) ≈ pit_vals
        @test !(loo_pit(y, y_pred .+ ϵ, log_weights) ≈ pit_vals)
        @test loo_pit(y, y_pred .+ ϵ, log_weights; is_discrete=true) ≈ pit_vals
    end
    @testset "DimArray data" begin
        draw_dim = Dim{:draw}(1:100)
        chain_dim = Dim{:chain}(0:2)
        sample_dims = (draw_dim, chain_dim)
        param_dims = (Dim{:param1}(1:2), Dim{:param2}([:a, :b, :c]))
        all_dims = (sample_dims..., param_dims...)
        y = DimArray(randn(size(param_dims)...), param_dims)
        y_pred = DimArray(randn(size(all_dims)...), all_dims)
        weights = DimArray(rand(size(all_dims)...), all_dims)
        weights ./= sum(weights; dims=(:draw, :chain))
        log_weights = log.(weights)
        pitvals = @inferred loo_pit(y, y_pred, log_weights)
        @test pitvals isa typeof(y)
        @test all(p -> 0 ≤ p ≤ 1, pitvals)
        @test DimensionalData.data(pitvals) ==
            loo_pit(map(DimensionalData.data, (y, y_pred, log_weights))...)
    end
    @testset "from InferenceData" begin
        draw_dim = Dim{:draw}(1:100)
        chain_dim = Dim{:chain}(0:2)
        sample_dims = (draw_dim, chain_dim)
        param_dims = (Dim{:param1}(1:2), Dim{:param2}([:a, :b, :c]))
        all_dims = (sample_dims..., param_dims...)
        y = DimArray(randn(size(param_dims)...), param_dims)
        z = DimArray(fill(randn()), ())
        y_pred = DimArray(randn(size(all_dims)...), all_dims)
        log_like = DimArray(randn(size(all_dims)...), all_dims)
        log_weights = loo(log_like).psis_result.log_weights
        pit_vals = loo_pit(y, y_pred, log_weights)

        idata1 = InferenceData(;
            observed_data=Dataset((; y)),
            posterior_predictive=Dataset((; y=y_pred)),
            log_likelihood=Dataset((; y=log_like)),
        )
        @test_throws Exception loo_pit(idata1; y_name=:z)
        @test_throws Exception loo_pit(idata1; y_pred_name=:z)
        @test_throws Exception loo_pit(idata1; log_likelihood_name=:z)
        @test loo_pit(idata1) == pit_vals
        VERSION ≥ v"1.7" && @inferred loo_pit(idata1)
        @test loo_pit(idata1; y_name=:y) == pit_vals
        @test loo_pit(idata1; y_name=:y, y_pred_name=:y, log_likelihood_name=:y) == pit_vals

        idata2 = InferenceData(;
            observed_data=Dataset((; z, y)),
            posterior_predictive=Dataset((; y_pred)),
            log_likelihood=Dataset((; log_like)),
        )
        @test_throws ArgumentError loo_pit(idata2)
        @test_throws ArgumentError loo_pit(
            idata2; y_name=:z, y_pred_name=:y_pred, log_likelihood_name=:log_like
        )
        @test_throws ArgumentError loo_pit(idata2; y_name=:y, y_pred_name=:y_pred)
        @test loo_pit(idata2; y_name=:y, log_likelihood_name=:log_like) == pit_vals
        @test loo_pit(
            idata2; y_name=:y, y_pred_name=:y_pred, log_likelihood_name=:log_like
        ) == pit_vals
        idata3 = InferenceData(;
            observed_data=Dataset((; y)),
            posterior_predictive=Dataset((; y=y_pred)),
            sample_stats=Dataset((; log_likelihood=log_like)),
        )
        @test loo_pit(idata3) == pit_vals
        VERSION ≥ v"1.7" && @inferred loo_pit(idata3)

        all_dims_perm = (param_dims..., reverse(sample_dims)...)
        idata4 = InferenceData(;
            observed_data=Dataset((; y)),
            posterior_predictive=Dataset((; y=permutedims(y_pred, all_dims_perm))),
            log_likelihood=Dataset((; y=permutedims(log_like, all_dims_perm))),
        )
        @test loo_pit(idata4) ≈ pit_vals
        VERSION ≥ v"1.7" && @inferred loo_pit(idata4)

        idata5 = InferenceData(;
            observed_data=Dataset((; y)), posterior_predictive=Dataset((; y=y_pred))
        )
        @test_throws ArgumentError loo_pit(idata5)
    end
end
