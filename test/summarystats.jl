using IteratorInterfaceExtensions
using OrderedCollections
using PosteriorStats
using Tables
using TableTraits
using Test

@testset "SummaryStats" begin
    labels = ["a", "bb", "ccc", "d", "e"]
    data = (est=randn(5), mcse_est=rand(5), rhat=rand(5), ess=rand(5))
    data_with_labels = merge((; label=labels), data)
    data_with_default_labels = merge((; label=Base.OneTo(5)), data)

    @testset "constructors" begin
        stats1 = SummaryStats(data)
        @test stats1.data == data
        @test isnothing(stats1.labels)
        @test stats1.name == "SummaryStats"

        stats2 = SummaryStats(data; name="Stats")
        @test stats2.data == data
        @test isnothing(stats2.labels)
        @test stats2.name == "Stats"

        stats3 = SummaryStats(data; labels)
        @test stats3.data == data
        @test stats3.labels == labels
        @test stats3.name == "SummaryStats"

        stats4 = SummaryStats(data; labels, name="Stats")
        @test stats4.data == data
        @test stats4.labels == labels
        @test stats4.name == "Stats"

        stats5 = SummaryStats(data_with_labels)
        @test stats5.data == data
        @test stats5.labels == labels

        stats6 = SummaryStats(merge(data, (; label=labels)))
        @test stats6.data == data
        @test stats6.labels == labels
        @test stats6.name == "SummaryStats"

        @test_throws ArgumentError SummaryStats(data_with_labels; labels)
        @test_throws DimensionMismatch SummaryStats(data; labels=labels[1:(end - 1)])
    end

    @inferred SummaryStats(data; name="Stats")
    stats_with_names(data, name, labels) = SummaryStats(data; name, labels)
    stats = @inferred stats_with_names(data, "Stats", labels)

    @testset "basic interfaces" begin
        @test parent(stats) == data
        @test stats.name == "Stats"
        @test SummaryStats(data; name="MoreStats").name == "MoreStats"
        @test keys(stats) == (:label, keys(data)...)
        for k in keys(stats)
            @test haskey(stats, k)
            @test getindex(stats, k) == getindex(data_with_labels, k)
        end
        @test !haskey(stats, :foo)
        @test length(stats) == length(data) + 1
        @test getindex(stats, 1) == labels
        for i in 1:length(data)
            @test stats[i + 1] == data[i]
        end
        @test Base.iterate(stats) == Base.iterate(data_with_labels)
        _, state = Base.iterate(stats)
        @test Base.iterate(stats, state) == Base.iterate(data_with_labels, state)

        data_copy1 = deepcopy(data)
        stats2 = SummaryStats(data_copy1; labels)
        @test stats2 == stats
        @test isequal(stats2, stats)

        data_copy2 = deepcopy(data)
        labels2 = copy(labels)
        labels2[1] = "foo"
        stats3 = SummaryStats(data_copy2; labels=labels2, name="Stats")
        @test stats3 != stats2
        @test !isequal(stats3, stats2)
        stats3 = SummaryStats(data_copy2; labels, name="Stats")
        stats3[:est][2] = NaN
        @test stats3 != stats2
        @test !isequal(stats3, stats2)
        stats2[:est][2] = NaN
        @test stats3 != stats2
        @test isequal(stats3, stats2)
    end

    @testset "merge" begin
        stats_dict = SummaryStats(OrderedDict(pairs(data)); labels, name="Stats")
        @test merge(stats) === stats
        @test merge(stats, stats) == stats
        @test merge(stats_dict) === stats_dict
        @test merge(stats_dict, stats_dict) == stats_dict
        @test merge(stats, stats_dict) == stats
        @test merge(stats_dict, stats) == stats_dict

        data2 = (ess=randn(5), rhat=rand(5), mcse_est=rand(5), est2=rand(5))
        stats2 = SummaryStats(data2; labels=1:5, name="Stats2")
        stats2_dict = SummaryStats(OrderedDict(pairs(data2)); labels=1:5, name="Stats2")
        for stats_a in (stats, stats_dict), stats_b in (stats2, stats2_dict)
            @test merge(stats_a, stats_b) ==
                SummaryStats(merge(data, data2); labels=stats_b[:label])
            @test merge(stats_a, stats_b).name == stats_b.name
            @test merge(stats_b, stats_a) ==
                SummaryStats(merge(data2, data); labels=stats_a[:label])
            @test merge(stats_b, stats_a).name == stats_a.name
        end
    end

    @testset "Tables interface" begin
        @test Tables.istable(typeof(stats))
        @test Tables.columnaccess(typeof(stats))
        @test Tables.columns(stats) === stats
        @test Tables.columnnames(stats) == keys(stats)
        table = Tables.columntable(stats)
        @test table == data_with_labels
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
            (; (k => Tables.getcolumn(stats, k)[1] for k in Tables.columnnames(stats))...),
        )
        nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(stats), 2))[2]
        @test isequal(
            nt,
            (; (k => Tables.getcolumn(stats, k)[2] for k in Tables.columnnames(stats))...),
        )
    end

    @testset "show" begin
        labels = ["a", "bb", "ccc", "d", "e"]
        data = (
            est=[111.11, 1.2345e-6, 5.4321e8, Inf, NaN],
            mcse_est=[0.0012345, 5.432e-5, 2.1234e5, Inf, NaN],
            se_est=vcat(0.0012345, 5.432e-5, 2.1234e5, Inf, NaN),
            rhat=vcat(1.009, 1.011, 0.99, Inf, NaN),
            rhat_bulk=vcat(1.009, 1.011, 0.99, Inf, NaN),
            ess=vcat(312.45, 23.32, 1011.98, Inf, NaN),
            ess_bulk=vcat(9.2345, 876.321, 999.99, Inf, NaN),
        )
        stats = SummaryStats(data; labels)

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
