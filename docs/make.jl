using PosteriorStats
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(PosteriorStats, :DocTestSetup, :(using PosteriorStats); recursive=true)

links = InterLinks(
    "IntervalSets" => (
        "https://juliamath.github.io/IntervalSets.jl/stable/",
        joinpath(@__DIR__, "inventories", "IntervalSets.toml"),
    ),
    "MCMCDiagnosticTools" => "https://julia.arviz.org/MCMCDiagnosticTools/stable/",
    "PSIS" => "https://julia.arviz.org/PSIS/stable/",
    "Statistics" => "https://docs.julialang.org/en/v1/",
    "StatsBase" => (
        "https://juliastats.org/StatsBase.jl/stable/",
        "https://juliastats.org/StatsBase.jl/dev/objects.inv",
    ),
)

makedocs(;
    modules=[PosteriorStats],
    repo=Remotes.GitHub("arviz-devs", "PosteriorStats.jl"),
    sitename="PosteriorStats.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true", edit_link="main", assets=String[]
    ),
    pages=["Home" => "index.md", "API" => "api.md"],
    warnonly=[:footnote, :missing_docs],
    plugins=[links],
)

deploydocs(; repo="github.com/arviz-devs/PosteriorStats.jl.git", devbranch="main")
