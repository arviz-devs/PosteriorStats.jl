using PosteriorStats
using Test

@testset "utilities for showing tables" begin
    @testset "_prettytables_sigdigits_formatter" begin
        @testset "all columns" begin
            @testset for sigdigits in 1:5
                ft1 = PosteriorStats._prettytables_sigdigits_formatter(sigdigits)
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
                ft = PosteriorStats._prettytables_sigdigits_formatter(sigdigits, [2, 3])
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

    @testset "_prettytables_sigdigits_from_se_formatter" begin
        data = (x=randn(10), x_se=rand(10), y=randn(10), y_se=rand(10))
        @testset for scale in 1:3, (col, se_col) in ((1, 2), (3, 4))
            ft = PosteriorStats._prettytables_sigdigits_from_se_formatter(data, 1, 2; scale)
            for i in eachindex(values(data)...), j in eachindex(data)
                v = data[j][i]
                if j == col
                    sigdigits = PosteriorStats.sigdigits_matching_se(
                        v, data[se_col][i]; scale
                    )
                    @test ft(v, i, j) == PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    @test ft("foo", i, j) == "foo"
                else
                    @test ft(v, i, j) === v
                end
            end
        end
    end
end
