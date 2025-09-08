using Logging: SimpleLogger, with_logger
using OffsetArrays
using PosteriorStats
using Test

@testset "waic" begin
    @testset "core functionality" begin
        @testset for sz in ((1000, 4), (1000, 4, 2), (100, 4, 2, 3)),
            T in (Float32, Float64),
            TA in (Array, OffsetArray)

            atol_perm = cbrt(eps(T))

            log_likelihood = randn(T, sz)
            if TA === OffsetArray
                log_likelihood = OffsetArray(log_likelihood, (0, -1, 10, 30)[1:length(sz)])
            end
            waic_result =
                TA === OffsetArrays ? waic(log_likelihood) : @inferred(waic(log_likelihood))
            @test waic_result isa PosteriorStats.WAICResult
            estimates = elpd_estimates(waic_result)
            pointwise = elpd_estimates(waic_result; pointwise=true)
            @testset "return types and values as expected" begin
                @test estimates isa NamedTuple{(:elpd, :se_elpd, :p, :se_p),NTuple{4,T}}
                @test pointwise isa NamedTuple{(:elpd, :p)}
                if length(sz) == 2
                    @test eltype(pointwise) === T
                else
                    @test eltype(pointwise) <: TA{T,length(sz) - 2}
                end
            end
            @testset "information criterion" begin
                @test information_criterion(waic_result, :log) == estimates.elpd
                @test information_criterion(waic_result, :negative_log) == -estimates.elpd
                @test information_criterion(waic_result, :deviance) == -2 * estimates.elpd
                @test information_criterion(waic_result, :log; pointwise=true) ==
                    pointwise.elpd
                @test information_criterion(waic_result, :negative_log; pointwise=true) ==
                    -pointwise.elpd
                @test information_criterion(waic_result, :deviance; pointwise=true) ==
                    -2 * pointwise.elpd
            end
        end
    end
    @testset "warnings" begin
        io = IOBuffer()
        log_likelihood = randn(100, 4)
        @testset for bad_val in (NaN, -Inf, Inf)
            log_likelihood[1] = bad_val
            result = with_logger(SimpleLogger(io)) do
                waic(log_likelihood)
            end
            msg = String(take!(io))
            @test occursin("Warning:", msg)
        end
    end
    @testset "show" begin
        loglike = log_likelihood_eight_schools().centered
        # regression test
        @test sprint(show, "text/plain", waic(loglike)) == """
            WAICResult with estimates
             elpd  se_elpd    p  se_p
              -31      1.4  0.9  0.32"""
    end
end
