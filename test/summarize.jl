using IteratorInterfaceExtensions
using MCMCDiagnosticTools
using OrderedCollections
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

_mean_and_std(x) = (mean=mean(x), std=std(x))

@testset "summary statistics" begin
    @testset "SummaryStats" begin
        parameter_names = ["a", "bb", "ccc", "d", "e"]
        data = (est=randn(5), mcse_est=rand(5), rhat=rand(5), ess=rand(5))

        @inferred SummaryStats(data; name="Stats")
        stats = @inferred SummaryStats(data, parameter_names; name="Stats")

        @testset "basic interfaces" begin
            @test parent(stats) === data
            @test stats.name == "Stats"
            @test SummaryStats("MoreStats", data).name == "MoreStats"
            @test SummaryStats(data; name="MoreStats").name == "MoreStats"
            @test keys(stats) == (:parameter, keys(data)...)
            for k in keys(stats)
                @test haskey(stats, k)
                if k === :parameter
                    @test getindex(stats, k) == parameter_names
                else
                    @test getindex(stats, k) == getindex(data, k)
                end
            end
            @test !haskey(stats, :foo)
            @test length(stats) == length(data) + 1
            @test getindex(stats, 1) == parameter_names
            for i in 1:length(data)
                @test stats[i + 1] == data[i]
            end
            @test Base.iterate(stats) == (parameter_names, (2, length(stats)))
            @test Base.iterate(stats, (2, length(stats))) == (stats[2], (3, length(stats)))

            data_copy1 = deepcopy(data)
            stats2 = SummaryStats(data_copy1, parameter_names)
            @test stats2 == stats
            @test isequal(stats2, stats)

            data_copy2 = deepcopy(data)
            parameter_names2 = copy(parameter_names)
            parameter_names2[1] = "foo"
            stats3 = SummaryStats(data_copy2, parameter_names2; name="Stats")
            @test stats3 != stats2
            @test !isequal(stats3, stats2)
            stats3 = SummaryStats(data_copy2, parameter_names; name="Stats")
            stats3[:est][2] = NaN
            @test stats3 != stats2
            @test !isequal(stats3, stats2)
            stats2[:est][2] = NaN
            @test stats3 != stats2
            @test isequal(stats3, stats2)
        end

        @testset "merge" begin
            stats_dict = SummaryStats(
                OrderedDict(pairs(data)), parameter_names; name="Stats"
            )
            @test merge(stats) === stats
            @test merge(stats, stats) == stats
            @test merge(stats_dict) === stats_dict
            @test merge(stats_dict, stats_dict) == stats_dict
            @test merge(stats, stats_dict) == stats
            @test merge(stats_dict, stats) == stats_dict

            data2 = (ess=randn(5), rhat=rand(5), mcse_est=rand(5), est2=rand(5))
            stats2 = SummaryStats(data2, 1:5; name="Stats2")
            stats2_dict = SummaryStats(OrderedDict(pairs(data2)), 1:5; name="Stats2")
            for stats_a in (stats, stats_dict), stats_b in (stats2, stats2_dict)
                @test merge(stats_a, stats_b) ==
                    SummaryStats(merge(data, data2), stats_b.parameter_names)
                @test merge(stats_a, stats_b).name == stats_b.name
                @test merge(stats_b, stats_a) ==
                    SummaryStats(merge(data2, data), stats_a.parameter_names)
                @test merge(stats_b, stats_a).name == stats_a.name
            end
        end

        @testset "Tables interface" begin
            @test Tables.istable(typeof(stats))
            @test Tables.columnaccess(typeof(stats))
            @test Tables.columns(stats) === stats
            @test Tables.columnnames(stats) == keys(stats)
            table = Tables.columntable(stats)
            @test table == (; parameter=parameter_names, data...)
            for (i, k) in enumerate(Tables.columnnames(stats))
                @test Tables.getcolumn(stats, i) == Tables.getcolumn(stats, k)
            end
            @test_throws ErrorException Tables.getcolumn(stats, :foo)
            @test !Tables.rowaccess(typeof(stats))
            @test Tables.schema(stats) == Tables.schema(Tables.columntable(stats))
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
            parameter_names = ["a", "bb", "ccc", "d", "e"]
            data = (
                est=[111.11, 1.2345e-6, 5.4321e8, Inf, NaN],
                mcse_est=[0.0012345, 5.432e-5, 2.1234e5, Inf, NaN],
                se_est=vcat(0.0012345, 5.432e-5, 2.1234e5, Inf, NaN),
                rhat=vcat(1.009, 1.011, 0.99, Inf, NaN),
                rhat_bulk=vcat(1.009, 1.011, 0.99, Inf, NaN),
                ess=vcat(312.45, 23.32, 1011.98, Inf, NaN),
                ess_bulk=vcat(9.2345, 876.321, 999.99, Inf, NaN),
            )
            stats = SummaryStats(data, parameter_names)
            @test sprint(show, "text/plain", stats) == """
                SummaryStats
                              est  mcse_est   se_est  rhat  rhat_bulk   ess  ess_bulk
                 a    111.110       0.0012   0.0012   1.01       1.01   312         9
                 bb     1e-06       5.4e-05  5.4e-05  1.01       1.01    23       876
                 ccc    5.432e+08   2.1e+05  2.1e+05  0.99       0.99  1012      1000
                 d       Inf         Inf      Inf      Inf        Inf   Inf       Inf
                 e       NaN         NaN      NaN      NaN        NaN   NaN       NaN"""

            @test startswith(sprint(show, "text/html", stats), "<table")
        end
    end

    @testset "summarize" begin
        @testset "base cases" begin
            x = randn(1_000, 4, 3)
            if VERSION ≥ v"1.9.0-"
                stats1 = @inferred summarize(x, mean, std, median)
            else
                stats1 = summarize(x, mean, std, median)
            end
            @test stats1 isa SummaryStats
            @test getfield(stats1, :name) == "SummaryStats"
            @test stats1 == SummaryStats(
                (
                    mean=map(mean, eachslice(x; dims=3)),
                    std=map(std, eachslice(x; dims=3)),
                    median=map(median, eachslice(x; dims=3)),
                ),
                axes(x, 3),
            )

            function _compute_stats(x)
                return summarize(x, (:mean, :std) => mean_and_std, :median => median)
            end
            if VERSION ≥ v"1.10.0-"
                stats2 = @inferred _compute_stats(x)
            else
                stats2 = _compute_stats(x)
            end
            @test stats2 == stats1

            stats3 = summarize(x, mean, std; var_names=["a", "b", "c"], name="Stats")
            @test getfield(stats3, :name) == "Stats"
            @test stats3 == SummaryStats(
                (mean=map(mean, eachslice(x; dims=3)), std=map(std, eachslice(x; dims=3))),
                ["a", "b", "c"],
            )

            stats4 = summarize(x; var_names=["a", "b", "c"], name="Stats")
            @test getfield(stats4, :name) == "Stats"
            stats5 = summarize(
                x, default_summary_stats()...; var_names=["a", "b", "c"], name="Stats"
            )
            @test stats4 == stats5

            stats6 = summarize(x, _mean_and_std, mad)
            @test haskey(stats6, :mean)
            @test haskey(stats6, :std)
            @test haskey(stats6, :mad)

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
                        _isapprox,
                        stats1,
                        summarize(
                            x,
                            mean,
                            std,
                            Symbol("hdi_94%") => hdi,
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
                        _isapprox,
                        stats2,
                        summarize(
                            x,
                            median,
                            mad,
                            Symbol("eti_90%") =>
                                (x -> PosteriorStats.eti(vec(x); prob=0.9)),
                            :mcse_median => (x -> mcse(x; kind=median)),
                            :ess_tail => (x -> ess(x; kind=:tail)),
                            :ess_median => (x -> ess(x; kind=median)),
                            rhat,
                        ),
                    ),
                )
                _compute_diagnostics(x) = summarize(x, default_diagnostics()...)
                if VERSION ≥ v"1.10.0-"
                    stats3 = @inferred _compute_diagnostics(x)
                else
                    stats3 = _compute_diagnostics(x)
                end
                @test all(
                    map(
                        _isapprox,
                        stats3,
                        summarize(
                            x,
                            :mcse_mean => mcse,
                            :mcse_std => (x -> mcse(x; kind=std)),
                            :ess_tail => (x -> ess(x; kind=:tail)),
                            :ess_bulk => (x -> ess(x; kind=:bulk)),
                            rhat,
                        ),
                    ),
                )

                @test all(
                    map(
                        _isapprox,
                        summarize(x, default_stats()...),
                        summarize(x, mean, std, Symbol("hdi_94%") => hdi),
                    ),
                )

                x2 = convert(Array{Union{Float64,Missing}}, x)
                x2[1, 1, 1] = missing
                stats4 = summarize(x2, default_summary_stats()...)
                @test stats4[:mean] ≈ [mean(skipmissing(x2[:, :, 1])); stats1[:mean][2:end]]
                @test stats4[:std] ≈ [std(skipmissing(x2[:, :, 1])); stats1[:std][2:end]]
                @test stats4[Symbol("hdi_94%")] == [
                    hdi(collect(skipmissing(x2[:, :, 1])))
                    stats1[Symbol("hdi_94%")][2:end]
                ]
                for k in (:mcse_mean, :mcse_std, :ess_tail, :ess_bulk, :rhat)
                    @test stats4[k][1] === missing
                    @test stats4[k][2:end] ≈ stats1[k][2:end]
                end

                stats5 = summarize(x2, default_summary_stats(median; prob_interval=0.9)...)
                @test stats5[:median] ≈
                    [median(skipmissing(x2[:, :, 1])); stats2[:median][2:end]]
                @test stats5[:mad] ≈ [mad(skipmissing(x2[:, :, 1])); stats2[:mad][2:end]]
                @test stats5[Symbol("eti_90%")] == [
                    PosteriorStats.eti(collect(skipmissing(x2[:, :, 1])); prob=0.9)
                    stats2[Symbol("eti_90%")][2:end]
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
                if VERSION ≥ v"1.10.0-"
                    stats1 = @inferred _compute_diagnostics(sample)
                else
                    stats1 = _compute_diagnostics(sample)
                end
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
