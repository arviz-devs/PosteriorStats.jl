using IteratorInterfaceExtensions
using MCMCDiagnosticTools
using PosteriorStats
using Statistics
using StatsBase
using Tables
using TableTraits
using Test

struct SampleWrapper{T,N}
    draws::T
    var_names::N
end

function PosteriorStats.summarize(
    spl::SampleWrapper, stats_funs...; var_names=spl.var_names, kwargs...
)
    x = spl.draws
    return summarize(x, stats_funs...; var_names, kwargs...)
end

@testset "summary statistics" begin
    @testset "SummaryStats" begin
        data = (
            parameter=["a", "bb", "ccc", "d", "e"],
            est=randn(5),
            mcse_est=randn(5),
            rhat=rand(5),
            ess=rand(5),
        )

        stats = @inferred SummaryStats(data; name="Stats")

        @testset "basic interfaces" begin
            @test parent(stats) === data
            @test stats.name == "Stats"
            @test SummaryStats("MoreStats", data).name == "MoreStats"
            @test SummaryStats(data; name="MoreStats").name == "MoreStats"
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

            stats2 = SummaryStats((; est=randn(5), est2=randn(5)); name="MoreStats")
            @test parent(stats2).parameter == 1:5
            stats_merged1 = merge(stats, stats2)
            @test stats_merged1.name == "Stats"
            @test parent(stats_merged1) == merge(parent(stats), parent(stats2))

            stats_merged2 = merge(stats2, stats)
            @test stats_merged2.name == "MoreStats"
            @test parent(stats_merged2) == merge(parent(stats2), parent(stats))
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
                parameter=["a", "bb", "ccc", "d", "e"],
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

    @testset "summarize" begin
        @testset "base cases" begin
            x = randn(1_000, 4, 3)
            stats1 = @inferred summarize(x, mean, std, median)
            @test stats1 isa SummaryStats
            @test getfield(stats1, :name) == "SummaryStats"
            @test stats1 == SummaryStats((
                parameter=axes(x, 3),
                mean=map(mean, eachslice(x; dims=3)),
                std=map(std, eachslice(x; dims=3)),
                median=map(median, eachslice(x; dims=3)),
            ))

            function _compute_stats(x)
                return summarize(x, (:mean, :std) => mean_and_std, :median => median)
            end
            stats2 = @inferred _compute_stats(x)
            @test stats2 == stats1

            stats3 = summarize(x, mean, std; var_names=["a", "b", "c"], name="Stats")
            @test getfield(stats3, :name) == "Stats"
            @test stats3 == SummaryStats((
                parameter=["a", "b", "c"],
                mean=map(mean, eachslice(x; dims=3)),
                std=map(std, eachslice(x; dims=3)),
            ))

            stats4 = summarize(x; var_names=["a", "b", "c"], name="Stats")
            @test getfield(stats4, :name) == "Stats"
            stats5 = summarize(
                x, default_summary_stats()...; var_names=["a", "b", "c"], name="Stats"
            )
            @test stats4 == stats5

            @test_throws DimensionMismatch summarize(x, mean; var_names=["a", "b"])
            @test_throws ArgumentError summarize(x, "std" => std)
            @test_throws ArgumentError summarize(x, ("mean", "std") => mean_and_std)
        end

        @testset "default stats function sets" begin
            @testset "array inputs" begin
                x = randn(1_000, 4, 3)
                # not completely type-inferrable due to HDI
                stats1 = summarize(x, default_summary_stats()...)
                @test all(
                    map(
                        ≈,
                        stats1,
                        summarize(
                            x,
                            mean,
                            std,
                            (Symbol("hdi_3%"), Symbol("hdi_97%")) => hdi,
                            :mcse_mean => mcse,
                            :mcse_std => (x -> mcse(x; kind=std)),
                            :ess_tail => (x -> ess(x; kind=:tail)),
                            :ess_bulk => (x -> ess(x; kind=:bulk)),
                            rhat,
                        ),
                    ),
                )
                stats2 = summarize(x, default_summary_stats(median; prob_interval=0.9)...)
                @test all(
                    map(
                        ≈,
                        stats2,
                        summarize(
                            x,
                            median,
                            mad,
                            (Symbol("eti_5%"), Symbol("eti_95%")) =>
                                (x -> quantile(x, (0.05, 0.95))),
                            :mcse_median => (x -> mcse(x; kind=median)),
                            :ess_tail => (x -> ess(x; kind=:tail)),
                            :ess_median => (x -> ess(x; kind=median)),
                            rhat,
                        ),
                    ),
                )
                _compute_diagnostics(x) = summarize(x, default_diagnostics()...)
                stats3 = @inferred _compute_diagnostics(x)
                @test stats3 == summarize(
                    x,
                    :mcse_mean => mcse,
                    :mcse_std => (x -> mcse(x; kind=std)),
                    :ess_tail => (x -> ess(x; kind=:tail)),
                    :ess_bulk => (x -> ess(x; kind=:bulk)),
                    rhat,
                )

                x2 = convert(Array{Union{Float64,Missing}}, x)
                x2[1, 1, 1] = missing
                stats4 = summarize(x2, default_summary_stats()...)
                @test stats4[:mean] ≈ [mean(skipmissing(x2[:, :, 1])); stats1[:mean][2:end]]
                @test stats4[:std] ≈ [std(skipmissing(x2[:, :, 1])); stats1[:std][2:end]]
                @test stats4[Symbol("hdi_3%")] ≈ [
                    hdi(collect(skipmissing(x2[:, :, 1]))).lower
                    stats1[Symbol("hdi_3%")][2:end]
                ]
                @test stats4[Symbol("hdi_97%")] ≈ [
                    hdi(collect(skipmissing(x2[:, :, 1]))).upper
                    stats1[Symbol("hdi_97%")][2:end]
                ]
                for k in (:mcse_mean, :mcse_std, :ess_tail, :ess_bulk, :rhat)
                    @test stats4[k][1] === missing
                    @test stats4[k][2:end] ≈ stats1[k][2:end]
                end

                stats5 = summarize(x2, default_summary_stats(median; prob_interval=0.9)...)
                @test stats5[:median] ≈
                    [median(skipmissing(x2[:, :, 1])); stats2[:median][2:end]]
                @test stats5[:mad] ≈ [mad(skipmissing(x2[:, :, 1])); stats2[:mad][2:end]]
                @test stats5[Symbol("eti_5%")] ≈ [
                    quantile(skipmissing(x2[:, :, 1]), 0.05)
                    stats2[Symbol("eti_5%")][2:end]
                ]
                @test stats5[Symbol("eti_95%")] ≈ [
                    quantile(skipmissing(x2[:, :, 1]), 0.95)
                    stats2[Symbol("eti_95%")][2:end]
                ]
                for k in (:mcse_median, :ess_tail, :ess_median, :rhat)
                    @test stats5[k][1] === missing
                    @test stats5[k][2:end] ≈ stats2[k][2:end]
                end
            end

            @testset "custom inputs" begin
                x = randn(1_000, 4, 3)
                sample = SampleWrapper(x, ["a", "b", "c"])
                function _compute_diagnostics(x)
                    return summarize(x, default_diagnostics()...; name="foo")
                end
                stats1 = @inferred _compute_diagnostics(sample)
                @test stats1 isa SummaryStats
                @test stats1.name == "foo"
                @test stats1 == summarize(
                    x, default_diagnostics()...; name="foo", var_names=["a", "b", "c"]
                )

                stats2 = summarize(sample)
                @test stats2.name == "SummaryStats"
                @test stats2 == summarize(x; var_names=["a", "b", "c"])
            end
        end
    end
end
