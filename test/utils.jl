using OffsetArrays
using PosteriorStats
using Random
using StatsBase
using Statistics
using Test

@testset "utils" begin
    @testset "_assimilar" begin
        @testset for x in ([8, 2, 5], (8, 2, 5), (; a=8, b=2, c=5))
            @test @inferred(PosteriorStats._assimilar((x=1.0, y=2.0, z=3.0), x)) ==
                (x=8, y=2, z=5)
            @test @inferred(PosteriorStats._assimilar((randn(3)...,), x)) == (8, 2, 5)
            y = OffsetVector(randn(3), -1)
            @test @inferred(PosteriorStats._assimilar(y, x)) == OffsetVector([8, 2, 5], -1)
        end
    end

    @testset "_sortperm/_permute" begin
        @testset for (x, y) in (
            [3, 1, 4, 2] => [1, 2, 3, 4],
            (3, 1, 4, 2) => (1, 2, 3, 4),
            (x=3, y=1, z=4, w=2) => (y=1, w=2, x=3, z=4),
        )
            perm = PosteriorStats._sortperm(x)
            @test perm == [2, 4, 1, 3]
            @test PosteriorStats._permute(x, perm) == y
        end
    end

    @testset "_logabssubexp" begin
        x, y = rand(2)
        @test @inferred(PosteriorStats._logabssubexp(log(x), log(y))) ≈ log(abs(x - y))
        @test PosteriorStats._logabssubexp(log(y), log(x)) ≈ log(abs(y - x))
    end

    @testset "_sum_and_se" begin
        @testset for n in (100, 1_000), scale in (1, 5)
            x = randn(n) * scale
            s, se = @inferred PosteriorStats._sum_and_se(x)
            @test s ≈ sum(x)
            @test se ≈ StatsBase.sem(x) * n

            x = randn(n, 10) * scale
            s, se = @inferred PosteriorStats._sum_and_se(x; dims=1)
            @test s ≈ sum(x; dims=1)
            @test se ≈ mapslices(StatsBase.sem, x; dims=1) * n

            x = randn(10, n) * scale
            s, se = @inferred PosteriorStats._sum_and_se(x; dims=2)
            @test s ≈ sum(x; dims=2)
            @test se ≈ mapslices(StatsBase.sem, x; dims=2) * n
        end
        @testset "::Number" begin
            @test isequal(PosteriorStats._sum_and_se(2), (2, NaN))
            @test isequal(PosteriorStats._sum_and_se(3.5f0; dims=()), (3.5f0, NaN32))
        end
    end

    @testset "_log_mean" begin
        x = rand(1000)
        logx = log.(x)
        w = rand(1000)
        w ./= sum(w)
        logw = log.(w)
        @test PosteriorStats._log_mean(logx, logw) ≈ log(mean(x, StatsBase.fweights(w)))
        x = rand(1000, 4)
        logx = log.(x)
        @test PosteriorStats._log_mean(logx, logw; dims=1) ≈
            log.(mean(x, StatsBase.fweights(w); dims=1))
    end

    @testset "_se_log_mean" begin
        ndraws = 1_000
        @testset for n in (1_000, 10_000), scale in (1, 5)
            x = rand(n) * scale
            w = rand(n)
            w = StatsBase.weights(w ./ sum(w))
            logx = log.(x)
            logw = log.(w)
            se = @inferred PosteriorStats._se_log_mean(logx, logw)
            se_exp = std(log(mean(rand(n) * scale, w)) for _ in 1:ndraws)
            @test se ≈ se_exp rtol = 1e-1
        end
    end

    @testset "sigdigits_matching_se" begin
        @test PosteriorStats.sigdigits_matching_se(123.456, 0.01) == 5
        @test PosteriorStats.sigdigits_matching_se(123.456, 1) == 3
        @test PosteriorStats.sigdigits_matching_se(123.456, 0.0001) == 7
        @test PosteriorStats.sigdigits_matching_se(1e5, 0.1) == 7
        @test PosteriorStats.sigdigits_matching_se(1e5, 0.2; scale=5) == 6
        @test PosteriorStats.sigdigits_matching_se(1e4, 0.5) == 5
        @test PosteriorStats.sigdigits_matching_se(1e4, 0.5; scale=1) == 6
        @test PosteriorStats.sigdigits_matching_se(1e5, 0.1; sigdigits_max=2) == 2

        # errors
        @test_throws ArgumentError PosteriorStats.sigdigits_matching_se(123.456, -1)
        @test_throws ArgumentError PosteriorStats.sigdigits_matching_se(
            123.456, 1; sigdigits_max=-1
        )
        @test_throws ArgumentError PosteriorStats.sigdigits_matching_se(
            123.456, 1; scale=-1
        )

        # edge cases
        @test PosteriorStats.sigdigits_matching_se(0.0, 1) == 0
        @test PosteriorStats.sigdigits_matching_se(NaN, 1) == 0
        @test PosteriorStats.sigdigits_matching_se(Inf, 1) == 0
        @test PosteriorStats.sigdigits_matching_se(100, 1; scale=Inf) == 0
        @test PosteriorStats.sigdigits_matching_se(100, Inf) == 0
        @test PosteriorStats.sigdigits_matching_se(100, 0) == 7
        @test PosteriorStats.sigdigits_matching_se(100, 0; sigdigits_max=2) == 2
    end

    @testset "_printf_with_sigdigits" begin
        @test PosteriorStats._printf_with_sigdigits(123.456, 1) == "1e+02"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 1) == "-1e+02"
        @test PosteriorStats._printf_with_sigdigits(123.456, 2) == "1.2e+02"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 2) == "-1.2e+02"
        @test PosteriorStats._printf_with_sigdigits(123.456, 3) == "123"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 3) == "-123"
        @test PosteriorStats._printf_with_sigdigits(123.456, 4) == "123.5"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 4) == "-123.5"
        @test PosteriorStats._printf_with_sigdigits(123.456, 5) == "123.46"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 5) == "-123.46"
        @test PosteriorStats._printf_with_sigdigits(123.456, 6) == "123.456"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 6) == "-123.456"
        @test PosteriorStats._printf_with_sigdigits(123.456, 7) == "123.4560"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 7) == "-123.4560"
        @test PosteriorStats._printf_with_sigdigits(123.456, 8) == "123.45600"
        @test PosteriorStats._printf_with_sigdigits(0.00000123456, 1) == "1e-06"
        @test PosteriorStats._printf_with_sigdigits(0.00000123456, 2) == "1.2e-06"
    end

    @testset "ft_printf_sigdigits" begin
        @testset "all columns" begin
            @testset for sigdigits in 1:5
                ft1 = PosteriorStats.ft_printf_sigdigits(sigdigits)
                for i in 1:10, j in 1:5
                    v = randn()
                    @test ft1(v, i, j) ==
                        PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    @test ft1("foo", i, j) == "foo"
                end
            end
        end
        @testset "subset of columns" begin
            @testset for sigdigits in 1:5
                ft = PosteriorStats.ft_printf_sigdigits(sigdigits, [2, 3])
                for i in 1:10, j in 1:5
                    v = randn()
                    if j ∈ [2, 3]
                        @test ft(v, i, j) ==
                            PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    else
                        @test ft(v, i, j) === v
                    end
                    @test ft("foo", i, j) == "foo"
                end
            end
        end
    end

    @testset "ft_printf_sigdigits_matching_se" begin
        @testset "all columns" begin
            @testset for scale in 1:3
                se = rand(5)
                ft = PosteriorStats.ft_printf_sigdigits_matching_se(se; scale)
                for i in eachindex(se), j in 1:5
                    v = randn()
                    sigdigits = PosteriorStats.sigdigits_matching_se(v, se[i]; scale)
                    @test ft(v, i, j) == PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    @test ft("foo", i, j) == "foo"
                end
            end
        end

        @testset "subset of columns" begin
            @testset for scale in 1:3
                se = rand(5)
                ft = PosteriorStats.ft_printf_sigdigits_matching_se(se, [2, 3]; scale)
                for i in eachindex(se), j in 1:5
                    v = randn()
                    if j ∈ [2, 3]
                        sigdigits = PosteriorStats.sigdigits_matching_se(v, se[i]; scale)
                        @test ft(v, i, j) ==
                            PosteriorStats._printf_with_sigdigits(v, sigdigits)
                        @test ft("foo", i, j) == "foo"
                    else
                        @test ft(v, i, j) === v
                    end
                end
            end
        end
    end
end
