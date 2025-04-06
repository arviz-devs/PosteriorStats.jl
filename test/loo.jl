using Logging: SimpleLogger, with_logger
using OffsetArrays
using PosteriorStats
using Test

@testset "loo" begin
    @testset "core functionality" begin
        @testset for sz in ((1000, 4), (1000, 4, 2), (100, 4, 2, 3)),
            T in (Float32, Float64),
            TA in (Array, OffsetArray)

            atol_perm = cbrt(eps(T))

            log_likelihood = randn(T, sz)
            if TA === OffsetArray
                log_likelihood = OffsetArray(log_likelihood, (0, -1, 10, 30)[1:length(sz)])
            end
            loo_result =
                TA === OffsetArray ? loo(log_likelihood) : @inferred(loo(log_likelihood))
            @test loo_result isa PosteriorStats.PSISLOOResult
            estimates = elpd_estimates(loo_result)
            pointwise = elpd_estimates(loo_result; pointwise=true)
            @testset "return types and values as expected" begin
                @test estimates isa NamedTuple{(:elpd, :elpd_mcse, :p, :p_mcse),NTuple{4,T}}
                @test pointwise isa
                    NamedTuple{(:elpd, :elpd_mcse, :p, :reff, :pareto_shape)}
                if length(sz) == 2
                    @test eltype(pointwise) === T
                else
                    @test eltype(pointwise) <: TA{T,length(sz) - 2}
                end
                @test loo_result.psis_result isa PSIS.PSISResult
                @test loo_result.psis_result.reff == pointwise.reff
                @test loo_result.psis_result.pareto_shape == pointwise.pareto_shape
            end
            @testset "information criterion" begin
                @test information_criterion(loo_result, :log) == estimates.elpd
                @test information_criterion(loo_result, :negative_log) == -estimates.elpd
                @test information_criterion(loo_result, :deviance) == -2 * estimates.elpd
                @test information_criterion(loo_result, :log; pointwise=true) ==
                    pointwise.elpd
                @test information_criterion(loo_result, :negative_log; pointwise=true) ==
                    -pointwise.elpd
                @test information_criterion(loo_result, :deviance; pointwise=true) ==
                    -2 * pointwise.elpd
            end
        end
    end
    # @testset "keywords forwarded" begin
    #     log_likelihood = convert_to_dataset((x=randn(1000, 4, 2, 3), y=randn(1000, 4, 3)))
    #     @test loo(log_likelihood; var_name=:x).estimates == loo(log_likelihood.x).estimates
    #     @test loo(log_likelihood; var_name=:y).estimates == loo(log_likelihood.y).estimates
    #     @test loo(log_likelihood; var_name=:x, reff=0.5).pointwise.reff == fill(0.5, 2, 3)
    # end
    # @testset "errors" begin
    #     log_likelihood = convert_to_dataset((x=randn(1000, 4, 2, 3), y=randn(1000, 4, 3)))
    #     @test_throws ArgumentError loo(log_likelihood)
    #     @test_throws ArgumentError loo(log_likelihood; var_name=:z)
    #     @test_throws DimensionMismatch loo(log_likelihood; var_name=:x, reff=rand(2))
    # end
    @testset "warnings" begin
        io = IOBuffer()
        log_likelihood = randn(100, 4)
        @testset for bad_val in (NaN, -Inf, Inf)
            log_likelihood[1] = bad_val
            result = with_logger(SimpleLogger(io)) do
                loo(log_likelihood)
            end
            msg = String(take!(io))
            @test occursin("Warning:", msg)
        end

        io = IOBuffer()
        log_likelihood = randn(100, 4)
        @testset for bad_reff in (NaN, 0, Inf)
            result = with_logger(SimpleLogger(io)) do
                loo(log_likelihood; reff=bad_reff)
            end
            msg = String(take!(io))
            @test occursin("Warning:", msg)
        end

        io = IOBuffer()
        log_likelihood = randn(5, 1)
        result = with_logger(SimpleLogger(io)) do
            loo(log_likelihood)
        end
        msg = String(take!(io))
        @test occursin("Warning:", msg)
    end
    @testset "show" begin
        loglike = log_likelihood_eight_schools(eight_schools_data().centered)
        # regression test
        @test sprint(show, "text/plain", loo(loglike)) == """
            PSISLOOResult with estimates
             elpd  elpd_mcse    p  p_mcse
              -31        1.4  0.9    0.33

            and PSISResult with 500 draws, 4 chains, and 8 parameters
            Pareto shape (k) diagnostic values:
                                Count      Min. ESS
             (-Inf, 0.5]  good  4 (50.0%)  270
              (0.5, 0.7]  okay  4 (50.0%)  307"""
    end
    @testset "agrees with R loo" begin
        if r_loo_installed()
            models = eight_schools_data()
            @testset for name in keys(models)
                log_likelihood = log_likelihood_eight_schools(models[name])
                reff_rand = rand(size(log_likelihood, 3))
                @testset for reff in (nothing, reff_rand)
                    result_r = loo_r(log_likelihood; reff)
                    result = loo(log_likelihood; reff)
                    @test result.estimates.elpd ≈ result_r.estimates.elpd
                    @test result.estimates.elpd_mcse ≈ result_r.estimates.elpd_mcse
                    @test result.estimates.p ≈ result_r.estimates.p
                    @test result.estimates.p_mcse ≈ result_r.estimates.p_mcse
                    @test result.pointwise.elpd ≈ result_r.pointwise.elpd
                    # increased tolerance for elpd_mcse, since we use a different approach
                    @test result.pointwise.elpd_mcse ≈ result_r.pointwise.elpd_mcse rtol =
                        0.01
                    @test result.pointwise.p ≈ result_r.pointwise.p
                    @test result.pointwise.reff ≈ result_r.pointwise.reff
                    @test result.pointwise.pareto_shape ≈ result_r.pointwise.pareto_shape
                end
            end
        else
            @warn "Skipping consistency tests against R loo::loo, since loo is not installed."
            @test_broken false
        end
    end
end
