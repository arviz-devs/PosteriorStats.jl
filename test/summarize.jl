using IteratorInterfaceExtensions
using MCMCDiagnosticTools
using PosteriorStats
using Statistics
using Tables
using TableTraits
using Test

@testset "summary statistics" begin
    @testset "SummaryStats" begin
        data = (
            variable=["a", "bb", "ccc", "d", "e"],
            est=randn(5),
            mcse_est=randn(5),
            rhat=rand(5),
            ess=rand(5),
        )

        stats = @inferred SummaryStats(data)

        @testset "basic interfaces" begin
            @test parent(stats) === data
            @test propertynames(stats) == propertynames(data)
            for k in propertynames(stats)
                @test getproperty(stats, k) == getproperty(data, k)
            end
            @test keys(stats) == keys(data)
            for k in keys(stats)
                @test haskey(stats, k) == haskey(data, k)
                @test getindex(stats, k) == getindex(data, k)
            end
            @test !haskey(stats, :foo)
            @test length(stats) == length(data)
            for i in 1:length(data)
                @test getindex(stats, i) == getindex(data, i)
            end
            @test Base.iterate(stats) == Base.iterate(data)
            @test Base.iterate(stats, 2) == Base.iterate(data, 2)
        end

        @testset "Tables interface" begin
            @test Tables.istable(typeof(stats))
            @test Tables.columnaccess(typeof(stats))
            @test Tables.columns(stats) === stats
            @test Tables.columnnames(stats) == keys(stats)
            table = Tables.columntable(stats)
            @test table == data
            for (i, k) in enumerate(Tables.columnnames(stats))
                @test Tables.getcolumn(stats, i) == Tables.getcolumn(stats, k)
            end
            @test_throws ErrorException Tables.getcolumn(stats, :foo)
            @test Tables.rowaccess(typeof(stats))
            @test Tables.rows(stats) == Tables.rows(parent(stats))
            @test Tables.schema(stats) == Tables.schema(parent(stats))
        end

        @testset "TableTraits interface" begin
            @test IteratorInterfaceExtensions.isiterable(stats)
            @test TableTraits.isiterabletable(stats)
            nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(stats), 1))[1]
            @test isequal(
                nt,
                (;
                    (
                        k => Tables.getcolumn(stats, k)[1] for
                        k in Tables.columnnames(stats)
                    )...
                ),
            )
            nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(stats), 2))[2]
            @test isequal(
                nt,
                (;
                    (
                        k => Tables.getcolumn(stats, k)[2] for
                        k in Tables.columnnames(stats)
                    )...
                ),
            )
        end

        @testset "show" begin
            data = (
                variable=["a", "bb", "ccc", "d", "e"],
                est=[111.11, 1.2345e-6, 5.4321e8, Inf, NaN],
                mcse_est=[0.0012345, 5.432e-5, 2.1234e5, Inf, NaN],
                rhat=vcat(1.009, 1.011, 0.99, Inf, NaN),
                ess=vcat(312.45, 23.32, 1011.98, Inf, NaN),
                ess_bulk=vcat(9.2345, 876.321, 999.99, Inf, NaN),
            )
            stats = SummaryStats(data)
            @test sprint(show, "text/plain", stats) == """
                SummaryStats
                              est  mcse_est  rhat   ess  ess_bulk
                 a    111.110       0.0012   1.01   312         9
                 bb     1.e-06      5.4e-05  1.01    23       876
                 ccc    5.432e+08   2.1e+05  0.99  1012      1000
                 d       Inf         Inf      Inf   Inf       Inf
                 e       NaN         NaN      NaN   NaN       NaN"""

            @test startswith(sprint(show, "text/html", stats), "<table")
        end
    end

    @testset "summarize" begin end
end
