using PosteriorStats
using Test

@testset "utilities for showing tables" begin
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
