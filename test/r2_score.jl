using GLM
using PosteriorStats
using Statistics
using Test

@testset "r2_score" begin
    @testset "basic" begin
        n = 100
        @testset for T in (Float32, Float64),
            sz in (300, (100, 3)),
            σ in T.((2, 1, 0.5, 0.1))

            x = range(T(0), T(1); length=n)
            slope = T(2)
            intercept = T(3)
            y = @. slope * x + intercept + randn(T) * σ
            x_reshape = length(sz) == 1 ? x' : reshape(x, 1, 1, :)
            y_pred = slope .* x_reshape .+ intercept .+ randn(T, sz..., n) .* σ

            r2_val = @inferred r2_score(y, y_pred; ci_prob=PosteriorStats.DEFAULT_CI_PROB)
            @test r2_val isa @NamedTuple{r2::T, eti::ClosedInterval{T}}
            r2_draws = @inferred PosteriorStats._r2_samples(y, y_pred)
            @test r2_val.r2 == mean(r2_draws)
            @test r2_val.eti == eti(r2_draws; prob=PosteriorStats.DEFAULT_CI_PROB)
            @test r2_val == r2_score(y, y_pred)

            r2_val2 = r2_score(
                y, y_pred; point_estimate=median, ci_fun=hdi, ci_prob=T(0.95)
            )
            @test r2_val2 isa @NamedTuple{r2::T, hdi::ClosedInterval{T}}
            @test r2_val2.r2 == median(r2_draws)
            @test r2_val2.hdi == hdi(r2_draws; prob=T(0.95))

            r2_draws2 = PosteriorStats.r2_score(y, y_pred; summary=false)
            @test r2_draws2 == r2_draws

            # check rough consistency with GLM
            res = lm(@formula(y ~ 1 + x), (; x=Float64.(x), y=Float64.(y)))
            @test r2_val.r2 ≈ r2(res) rtol = 1
        end
    end
end
